"""Tests for the ``remax._native`` compile-and-cache pipeline.

``tests/test_native.py`` and ``tests/test_hardening.py`` cover the *kernel
contract* — that the scan matches the LUT and that the argument validation
refuses anything that would hand the C code a bad pointer. A mutation pass
(``gating/scripts/mutate.py``, 34 sites) found that region well covered: 15 of
16 mutants killed. Everything upstream of it was not: 17 of 18 mutants in
``_cache_dir`` / ``_compile`` survived, including inverting the
compiler-success test on line 223. This file closes that gap.

What these tests still cannot catch
-----------------------------------
Stated because a green run is silent about the things it does not look at, in
exactly the tone it uses for the things it approved.

* **That the built library actually uses hardware POPCNT.** The flag-position
  test asserts ``-mpopcnt`` reaches the compiler in an argv position where it
  is an option and not the ``-o`` operand. It does not disassemble the result.
  A toolchain that accepts the flag and ignores it produces a correct, slower
  library and every test here passes. That is a benchmarking concern
  (``bench/native_speedup.py``), and it is currently in no CI job.
* **Anything platform-specific off the CI runner.** ``_lib_suffix`` returns
  ``.dylib`` on Darwin; single-OS CI cannot exercise the other branch. The
  ``os.getuid`` absence path in ``_cache_dir`` is likewise unreachable on
  POSIX — a mutant flipping its fallback value survives by construction.
* **A real compiler failing in a way the fake does not model.** The failure
  tests drive ``subprocess.run`` through a stub. A compiler that hangs past the
  30 s timeout, or that succeeds while emitting an unloadable object, is
  modelled here only as "returncode, plus whatever bytes landed at ``-o``".
* **Concurrency.** Atomic publish via ``os.replace`` is asserted structurally
  (no ``.tmp`` survives a completed call), not by racing two importers.
* **The cached-artifact trust boundary.** If a valid ``.so`` already sits at
  the hash-named path, ``_compile`` returns it without verifying it was built
  from ``_C_SOURCE``. The hash names the *source*, not the binary.

Surviving mutants, after this file
----------------------------------
The mutation pass over ``_native.py`` goes from 16/34 killed to 25/34. The nine
survivors are listed here so a future reader can tell a known hole from a new
one; re-run and diff against this list rather than against zero::

    python3 mutate.py --target src/remax/_native.py -- \
        python3 -m pytest tests/test_native.py tests/test_native_compile.py \
            tests/test_hardening.py tests/test_query_path.py \
            tests/test_query_speed_paths.py -q

* ``_SOURCE_HASH`` truncation length — a different prefix length is a different
  cache filename and nothing else. No behaviour to assert.
* The ``os.getuid`` fallback constant and the Darwin suffix branch — unreachable
  on a POSIX single-OS runner, per the limits above.
* ``capture_output`` / ``text`` / ``timeout`` on the ``subprocess.run`` call —
  the stub ignores its kwargs by construction. Killing these needs a real
  compiler invocation, which is what ``test_native.py`` already provides for the
  success path only.
* ``missing_ok`` in the failed-publish cleanup, and the two ``logger.debug``
  format arguments — inert: the temp file always exists on that branch, and a
  log line has no observable effect on the return value.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

from remax import _native


X86 = platform.machine() in ("x86_64", "AMD64")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A fresh, empty cache directory for _compile to work in."""
    d = tmp_path / "cache"
    monkeypatch.setenv("REMAX_CACHE_DIR", str(d))
    return d


def _fake_compiler(returncode, *, emit=b"", record=None):
    """Stand in for subprocess.run: write ``emit`` to the -o path, then exit.

    ``emit`` is deliberately independent of ``returncode`` — a compiler that
    leaves a partial object behind and *then* fails is the case that
    distinguishes a working success-check from an inverted one.
    """

    def run(cmd, **kwargs):
        if record is not None:
            record.append(list(cmd))
        if emit:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(emit)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="stub")

    return run


# ── compile failure must publish nothing ──────────────────────────────


def test_failed_compile_publishes_no_library(cache, monkeypatch):
    """A compiler that leaves output behind and *then* fails must not ship it.

    Known-bad: this is the case that goes red if the success test on the
    result of ``subprocess.run`` is inverted or weakened to ``or`` — both
    mutants publish the partial object as a working library.
    """
    monkeypatch.setattr(
        _native.subprocess, "run", _fake_compiler(1, emit=b"partial garbage")
    )

    assert _native._compile() is None

    published = list(cache.glob(f"*{_native._lib_suffix()}"))
    assert published == [], f"failed compile published {published}"


def test_failed_compile_leaves_no_temp_files(cache, monkeypatch):
    """Every attempt cleans up its own ``.tmp``, success or failure."""
    monkeypatch.setattr(
        _native.subprocess, "run", _fake_compiler(1, emit=b"partial garbage")
    )

    _native._compile()

    assert list(cache.glob("*.tmp")) == [], "temp objects survived a failed compile"


def test_missing_compiler_returns_none(cache, monkeypatch):
    """No gcc and no cc is a supported configuration, not a crash."""

    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(_native.subprocess, "run", run)

    assert _native._compile() is None
    assert list(cache.glob("*.tmp")) == []


def test_compiler_that_writes_nothing_cleans_up(cache, monkeypatch):
    """Failure with no output at all — cleanup must tolerate an absent temp file.

    Distinct from the partial-object case above: there the ``.tmp`` exists and
    must be removed, here it never existed and removing it must not raise.
    """
    monkeypatch.setattr(_native.subprocess, "run", _fake_compiler(1, emit=b""))

    assert _native._compile() is None
    assert list(cache.glob("*.tmp")) == []


def test_unusable_compiler_returns_none(cache, monkeypatch):
    """An OSError from the exec itself (ENOEXEC, EACCES) is a fallback, not a crash."""

    def run(cmd, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(_native.subprocess, "run", run)

    assert _native._compile() is None
    assert list(cache.glob("*.tmp")) == []


def test_failed_publish_returns_none(cache, monkeypatch):
    """If the atomic rename fails, report no library rather than a phantom path."""
    monkeypatch.setattr(_native.subprocess, "run", _fake_compiler(0, emit=b"\x7fELF"))

    def refuse(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(_native.os, "replace", refuse)

    assert _native._compile() is None
    assert list(cache.glob(f"*{_native._lib_suffix()}")) == []
    assert list(cache.glob("*.tmp")) == []


def test_out_buffer_is_keyword_only():
    """``out`` must stay keyword-only; positionally it would be read as a query."""
    import inspect

    sig = inspect.signature(_native.hamming_distances_native)
    assert sig.parameters["out"].kind is inspect.Parameter.KEYWORD_ONLY


def test_compiler_timeout_returns_none(cache, monkeypatch):
    """A hung compiler must fall back, not propagate."""

    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(_native.subprocess, "run", run)

    assert _native._compile() is None
    assert list(cache.glob("*.tmp")) == []


# ── the success bracket ───────────────────────────────────────────────


def test_successful_compile_publishes_once_and_privately(cache, monkeypatch):
    """The other edge: a clean exit must publish, atomically and 0o600.

    Paired with the failure test above, this brackets the success check —
    a one-sided assertion passes just as readily for a path that publishes
    everything as for one that publishes nothing.
    """
    monkeypatch.setattr(_native.subprocess, "run", _fake_compiler(0, emit=b"\x7fELF"))

    lib = _native._compile()

    assert lib is not None and lib.exists()
    assert lib.suffix == _native._lib_suffix()
    assert list(cache.glob("*.tmp")) == [], "publish was not atomic"
    assert os.stat(lib).st_mode & 0o777 == 0o600


def test_existing_library_is_reused_without_compiling(cache, monkeypatch):
    """A cache hit must not shell out at all."""
    monkeypatch.setattr(_native.subprocess, "run", _fake_compiler(0, emit=b"\x7fELF"))
    first = _native._compile()

    def refuse(cmd, **kwargs):
        raise AssertionError("recompiled despite a cache hit")

    monkeypatch.setattr(_native.subprocess, "run", refuse)
    assert _native._compile() == first


# ── compiler argv shape ───────────────────────────────────────────────


@pytest.mark.skipif(not X86, reason="-mpopcnt is x86-64 only")
def test_popcnt_flag_is_an_option_not_the_output_operand(cache, monkeypatch):
    """``-mpopcnt`` must land where the compiler reads it as a flag.

    It is spliced in by index. Off by one and it becomes the argument to
    ``-o``: the object is written to a file named ``-mpopcnt``, the real
    temp path never appears, and the only symptom is a fallback to NumPy —
    or, on a toolchain that tolerates it, a correct library built without
    hardware popcount and roughly 30x slower.
    """
    calls = []
    monkeypatch.setattr(
        _native.subprocess, "run", _fake_compiler(0, emit=b"\x7fELF", record=calls)
    )

    _native._compile()

    assert calls, "compiler was never invoked"
    cmd = calls[0]
    assert "-mpopcnt" in cmd
    o = cmd.index("-o")
    assert cmd[o + 1].endswith(".tmp"), (
        f"-o operand is {cmd[o + 1]!r}, not the temp object; "
        "a flag has been spliced into the wrong position"
    )
    assert cmd.index("-mpopcnt") < o, "-mpopcnt must precede -o"
    assert cmd[-1].endswith(".c"), "source path is no longer the final argument"


def test_optimisation_flag_survives(cache, monkeypatch):
    """-O3 is load-bearing: the docstring's 25-35x assumes it."""
    calls = []
    monkeypatch.setattr(
        _native.subprocess, "run", _fake_compiler(0, emit=b"\x7fELF", record=calls)
    )

    _native._compile()

    assert "-O3" in calls[0]
    assert "-shared" in calls[0] and "-fPIC" in calls[0]


# ── cache directory ───────────────────────────────────────────────────


def test_cache_dir_creates_missing_parents(tmp_path, monkeypatch):
    """REMAX_CACHE_DIR may point somewhere several levels deep and absent."""
    target = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("REMAX_CACHE_DIR", str(target))

    d = _native._cache_dir()

    assert d == target and d.is_dir()


def test_cache_dir_is_idempotent(cache):
    """Second call on an existing directory must not raise."""
    first = _native._cache_dir()
    second = _native._cache_dir()

    assert first == second == cache
    assert os.stat(second).st_mode & 0o777 == 0o700


def test_cache_dir_prefers_xdg_over_home(tmp_path, monkeypatch):
    monkeypatch.delenv("REMAX_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert _native._cache_dir() == tmp_path / "xdg" / "remax"


def test_cache_dir_falls_back_when_home_unresolvable(tmp_path, monkeypatch):
    """An unwritable/absent home must degrade to a per-user temp dir, still 0o700."""
    monkeypatch.delenv("REMAX_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        _native.Path, "home", classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError))
    )

    d = _native._cache_dir()

    assert d.is_dir()
    assert "remax_native" in d.name
    if hasattr(os, "getuid"):
        assert d.name.endswith(str(os.getuid())), (
            "fallback cache dir is not per-user; a co-located user could "
            "pre-place a library at a predictable path (CWE-379)"
        )
    assert os.stat(d).st_mode & 0o777 == 0o700
