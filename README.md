# War GPU Simulator

Massively parallel card game simulation using WebGPU compute shaders.

Simulates hundreds of thousands of games of War per second on the GPU, with smooth animated stats, record-breaking game detection, and round-by-round replay.

## How it works

- A WGSL compute shader runs 65,536 independent games per GPU dispatch
- Each game: Fisher-Yates shuffle → circular buffer card management → full War rules including multi-level wars
- Double-buffered pipeline: next dispatch runs while CPU processes current results
- Seeds are deterministic and sequential from a random base — every game is reproducible via replay

## Run locally

```bash
npm install
npm run dev
```

## Deploy to GitHub Pages

1. Update `base` in `vite.config.js` to match your repo name
2. Push to `main` — the GitHub Action builds and deploys automatically
3. Enable Pages in repo Settings → source: "GitHub Actions"

## Requirements

- A browser with WebGPU support (Chrome 113+, Edge, Firefox 141+, Safari 26+)
- A GPU (integrated is fine, discrete is faster)
