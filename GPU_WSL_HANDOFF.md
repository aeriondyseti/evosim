# GPU on WSL2 — Handoff / Continuation Doc

> Purpose: get **evosim** running on the **NVIDIA GPU via JAX inside WSL2**, then benchmark it
> against the CPU baseline. This doc is written so a fresh Claude Code session **started from
> inside WSL** (cwd `~/dev/evosim-framework`) can continue without the earlier context.

## TL;DR of what we're doing

evosim currently runs on **CPU JAX on native Windows** (no native Windows GPU build exists).
We're standing up the **WSL2 + CUDA** path so the same XLA code runs on the GPU. WSL2, the
NVIDIA driver, and CUDA passthrough are already verified working. What's left is the Python/JAX
setup **using mise to manage `uv`**, then verify the GPU is visible and benchmark.

## Environment facts (already verified)

- **OS**: Windows 11 host; WSL2 distro **Ubuntu-26.04** (default, version 2).
- **GPUs** (visible in WSL via `nvidia-smi -L`, driver **610.62**):
  - `GPU 0`: **RTX 5060 Ti, 16 GB** (Blackwell, `sm_120`) — most VRAM.
  - `GPU 1`: **RTX 4060 Ti, 8 GB** (Ada, `sm_89`) — rock-solid CUDA support.
- **CUDA passthrough works**: `nvidia-smi` runs inside WSL; `/usr/lib/wsl/lib/libcuda.so*` present.
  Do **NOT** install a Linux NVIDIA driver inside WSL — the Windows driver provides it.
- **Repo location (WSL)**: `~/dev/evosim-framework` (= `/home/aerion/dev/evosim-framework`).
  This is a **copy** of the canonical Windows checkout at `D:\Development\evosim-framework`
  (mirrored at `/mnt/d/Development/evosim-framework`). The Windows `.venv` was excluded from the copy.
- **CPU baseline to beat** (measured on Windows, 1,000,000-agent foragers):
  `foragers_large --headless-bench` → **~7.7 ticks/s ≈ 7.7M agent-ticks/s**; ~4 of 24 cores busy.

## Tooling decisions / constraints (read before acting)

1. **Use `mise` to manage `uv`** (mise-en-place is installed in this WSL).
   - A standalone `uv` was installed earlier via the curl installer at `~/.local/bin/uv`.
     Prefer mise's `uv`. If `which uv` points at `~/.local/bin/uv`, either remove it
     (`rm -f ~/.local/bin/uv ~/.local/bin/uvx`) or ensure mise's shims take precedence on PATH.
2. **Python must be 3.13, not 3.14.** The system python in WSL is 3.14, which JAX CUDA wheels
   don't support yet. The repo's `.python-version` pins **3.13** — keep it. Let `uv` (or mise)
   provide CPython 3.13.
3. **Never run `uv sync` against `/mnt/d/...`** — it would overwrite the Windows `.venv` with
   Linux binaries and is slow (9p FS). Always work in `~/dev/evosim-framework`.
4. **`gpu` extra already exists** in `pyproject.toml`:
   `gpu = ["jax[cuda12]>=0.10.2; sys_platform == 'linux'"]` (Linux-only via marker).
   `uv sync --extra gpu` pulls ~2–3 GB of CUDA wheels (jaxlib + nvidia-* runtime) — first run
   takes a few minutes.
5. **Blackwell note**: the RTX 5060 Ti is `sm_120`; it needs a recent CUDA (12.8+). The bundled
   `jax[cuda12]` CUDA should support it given the new driver, but if you hit an `sm_120` /
   "no kernel image" / PTX error, fall back to the 4060 Ti with `CUDA_VISIBLE_DEVICES=1`.

## Plan (do these in order, in `~/dev/evosim-framework`)

```bash
cd ~/dev/evosim-framework

# 1. Make mise provide uv (adapt to your mise setup; verify which uv afterwards).
mise use -g uv@latest        # or however this mise installs uv
hash -r; which uv; uv --version
#   If ~/.local/bin/uv shadows it:  rm -f ~/.local/bin/uv ~/.local/bin/uvx

# 2. Sync the GPU environment (creates a Linux .venv here; pulls CUDA wheels).
uv sync --extra gpu
#   (uv will update uv.lock for linux/cuda; that's expected.)

# 3. Verify JAX sees the GPU.
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
#   Expect default_backend == 'gpu' and a list of CudaDevice(s).

# 4. Benchmark on GPU and compare to the 7.7 ticks/s CPU baseline.
CUDA_VISIBLE_DEVICES=0 uv run python -m evosim.examples.foragers_large --headless-bench 50
#   GPU 0 = RTX 5060 Ti (16 GB). If it errors on sm_120, retry with CUDA_VISIBLE_DEVICES=1.

# 5. (Optional) Larger world now that there's VRAM headroom:
CUDA_VISIBLE_DEVICES=0 uv run python -m evosim.examples.foragers_large \
    --headless-bench 50 --agents 4000000 --grid 2048
```

## Success criteria

- `jax.default_backend()` returns `gpu`; `jax.devices()` lists the CUDA device(s).
- `foragers_large --headless-bench` runs on GPU and reports throughput **well above** the
  ~7.7 ticks/s CPU baseline (expect a large multiple at 1M agents).
- Full test suite still green on GPU: `uv run pytest` (set `SDL_VIDEODRIVER=dummy` if running the
  viewer tests headless; pygame from the `demos` extra is optional — add `--extra demos` if you
  want those too).

## Useful context pointers

- `SPEC.md` — full design; §11 perf targets, §13 notes the Windows-CPU / Linux-GPU split.
- `PROGRESS.md` — build ledger (library + 3 demos complete; viz + million-agent PoC added).
- `README.md` — quickstart, demos, the `foragers_large` rasterized scale PoC.
- Determinism is **same-device** bit-exact (counter-based RNG); CPU and GPU results will differ
  bit-for-bit (different reduction order) but should agree statistically — expected per SPEC.

## After GPU works (suggested follow-ups)

- Record GPU numbers in `PROGRESS.md` / `README.md` (replace the "validated on Linux/WSL2 later"
  caveats with real figures).
- Consider committing the linux `uv.lock` changes / GPU notes from the WSL copy back to the
  canonical Windows repo (or just keep the WSL copy as the GPU run environment).
- Multi-GPU: both cards are visible; `run_ensemble` could be `pmap`'d across them later.
