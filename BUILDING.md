# Building PufferLib

The native library (the `pufferlib._C` extension and standalone environment
executables) is built with CMake. `build.sh` (Linux/macOS) and `build.ps1`
(Windows) are thin wrappers over the presets in `CMakePresets.json`.

One environment is statically linked per build:

```bash
./build.sh breakout              # CUDA _C extension (default)
./build.sh breakout --float      # float32 precision (required for --slowly)
./build.sh breakout --cpu       # CPU-only extension, torch backend only
./build.sh breakout --debug      # Debug build
./build.sh breakout --local      # Standalone executable (debug, sanitizers)
./build.sh breakout --fast       # Standalone executable (optimized)
./build.sh all                   # Everything
```

Or use the presets directly: `cmake --preset cuda -DENV=breakout && cmake --build --preset cuda`.

Requirements (all platforms): CMake >= 3.24, Ninja, clang, Python with
`pybind11` and `numpy` installed (extension builds only). ccache is picked up
automatically when present. `compile_commands.json` is emitted into each
build directory for clangd users.

## CUDA dependencies

- **CUDA toolkit**: discovered via the standard `CUDA_PATH` environment
  variable (set by the NVIDIA installer), `nvcc` on PATH, or
  `-DCUDAToolkit_ROOT=<dir>`. GPU architecture defaults to `native`
  (the GPU in the build machine); override with `-DCMAKE_CUDA_ARCHITECTURES=89`
  or the legacy `NVCC_ARCH` env var through the wrappers.
- **cuDNN**: searched in `CUDNN_ROOT`/`CUDNN_PATH`, the CUDA toolkit dirs,
  then the `nvidia-cudnn-cu12` pip wheel (Linux). See the Windows notes below.
- **NCCL** (multi-GPU): system install or the `nvidia-nccl-cu12` wheel.
  Linux only; without it, single-GPU training works and multi-GPU raises a
  clear error at startup.

## Windows

Quickstart:

1. Install: Visual Studio 2022 Build Tools (C++ workload), LLVM (clang),
   CMake, Ninja, and the CUDA toolkit (12.x). Note: CUDA 12.x cannot use the
   VS2026 (v14.5x) toolset as nvcc host compiler; VS2022 Build Tools can be
   installed side by side with VS2026. CUDA >= 13.2 supports VS2026 directly.
2. For CUDA extension builds, cuDNN import libraries are required. Either
   install cuDNN from https://developer.nvidia.com/cudnn (set `CUDNN_ROOT` if
   installed to a custom location), or let the build download the NVIDIA
   redist archive once with `-FetchCudnn` (`-DPUFFER_FETCH_CUDNN=ON` via
   `cmake` directly; ~1.7 GB).
3. Build: `.\build.ps1 breakout` (add `-Cpu`, `-Fast`, `-Float`, `-DebugBuild`,
   `-Local`, `-FetchCudnn` as needed). The script locates and loads the right
   MSVC environment automatically (prefers VS2022 for CUDA builds).

Runtime DLLs: `pufferlib/__init__.py` registers the CUDA `bin` directory and
any NVIDIA pip-wheel `bin` directories on the DLL search path before loading
`_C`. PyTorch is imported before `_C` by the trainer and its bundled cuDNN
DLLs satisfy the extension, so no separate cuDNN runtime install is needed
when torch is present.

Windows notes:

- Multi-GPU is unavailable (NCCL is Linux-only); training is single-GPU.
- OpenMP is disabled inside the extension on Windows: torch's wheels ship
  Intel's OpenMP runtime (`libiomp5md.dll`) and loading LLVM's `libomp`
  alongside it aborts the process (OMP Error #15). Standalone executables
  do use OpenMP.

  Consequence (found 2026-07-24): the vecenv's intra-buffer env-step loop
  was an `omp parallel for`, so without OpenMP each buffer stepped all of
  its envs on ONE core — `[vec] num_threads` was a dead knob, CPU sat ~2%
  busy, and the GPU pipeline stalled behind a single-threaded simulator.
  Fix: `StaticEnvPool` in `src/vecenv.h`, a pthread-shim worker pool
  (atomic work-stealing cursor, generation-bump release, hard-spin waits)
  compiled only when `_OPENMP` is absent. Linux builds and standalone exes
  keep the OpenMP path. `num_threads / num_buffers` = env-step workers per
  buffer; on a 20-core box, `num_threads = 16` took starmelee from 1.6M to
  4.8M SPS (Env wall share 70% -> 43%, GPU util ~22% -> 65%).

  Alternatives considered for re-enabling real OpenMP in the extension
  (all build-only, none adopted):
  1. Link the extension against torch's own runtime: compile with
     `/openmp:llvm` (MSVC) or `-fopenmp` (clang) and link an import
     library generated from `torch/lib/libiomp5md.dll` (LLVM's libomp is
     ABI-compatible with Intel's — same `__kmpc_*` entry points). One
     runtime in-process, no Error #15. Cleanest option, but couples the
     build to torch's runtime-shipping choices.
  2. MSVC `/openmp` (vcomp140): a separate runtime family that does not
     trip Intel's duplicate detection and coexists with libiomp5md. Two
     thread pools in-process; OpenMP 2.0 only (sufficient here).
  3. `KMP_DUPLICATE_LIB_OK=TRUE`: documented-unsafe, ruled out.
  Decision: keep the pthread pool — measured equal-or-better (its dynamic
  cursor load-balances variable-cost env steps better than
  `schedule(static)`), zero runtime politics, portable.
- Environments that don't build on Windows yet: `nethack` (Linux syscalls),
  `chess` (fork/exec engine), `boxoban` (mmap), `impulse_wars` (no Windows
  box2d prebuilt), `trailer` (fork/wait), `onlyfish` standalone (dirent).
  Everything else in `ocean/` builds; a few standalone demos are broken
  upstream on all platforms (type mismatches caught by -Werror): `battle`,
  `blastar`, `convert`, `convert_circle`, `snake`, `whisker_racer`, and
  `matsci`/`shared_pool` need external libs (lammps, cpr).
- POSIX shims live in `src/puffer_os.h` (pthreads, clock_gettime, rand_r,
  rand/srand/RAND_MAX, usleep, aligned alloc, ...). It is force-included for
  env code on Windows. Note UCRT's `RAND_MAX` is 0x7FFF while the shimmed
  generators emit glibc's 31-bit range; the shim raises `RAND_MAX` to match,
  so `rand()/RAND_MAX` idioms behave like they do on Linux.
- The native CUDA trainer works on Windows (verified learning with CUDA
  graphs enabled). Historical note for future LLP64 debugging: it used to
  train at random level forever because `alloc_register(LongTensor*)` sized
  arena slots with `sizeof(long)` (4 bytes on Windows, 8 on Linux), letting
  the int64 RNG counters overwrite the adjacent `act_sizes` tensor every
  step. When adding `long`-typed sizes/offsets to the CUDA sources, use
  `int64_t` — Windows `long` is 32-bit.
- The PyTorch backend (`--slowly`, requires a `-Float` build) remains
  available as a slower cross-check for trainer-side debugging.

## Web builds (emscripten) — TODO

The `web` preset exists in `CMakePresets.json` and the CMake branch for it is
in place (mirrors the old `build.sh --web` emcc flags: ASYNCIFY, GLFW,
`vendor/minshell.html` shell, `resources/<env>` preloads, raylib webassembly
build). It has not been exercised yet. Remaining work:

1. Install/activate emsdk and verify `cmake --preset web -DENV=breakout`
   configures with `$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`.
2. Confirm `--preload-file` paths and output naming produce
   `build/web/<env>/game.html` byte-equivalent to the old script's output.
3. Envs without a `resources/<env>` directory need the preload made
   conditional.
4. Add a CI smoke job with `mymindstorm/setup-emsdk`.

## Profile builds

`./build.sh <env> --profile` builds `tests/profile_kernels.cu` into a
standalone `profile` binary (Linux only).
