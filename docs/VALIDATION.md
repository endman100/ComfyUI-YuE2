# Official versus ComfyUI validation

This report records the matched YuE2 acoustic-flow and decode comparison used for ComfyUI-YuE2.

## Tested revisions

| Component | Source | Revision |
| --- | --- | --- |
| Official YuE2 source | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | [`0edaf2f4053ef4731334b8329834b107977f9637`](https://github.com/multimodal-art-projection/YuE/commit/0edaf2f4053ef4731334b8329834b107977f9637) |
| YuE2-3B weights | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | [`1a96eca688d6ae5d7f0feb88573fec89920fcd19`](https://huggingface.co/m-a-p/YuE2-3B/commit/1a96eca688d6ae5d7f0feb88573fec89920fcd19) |
| YuE2 VAE weights | [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | `95535e72a97bc0f09b8ada125d26b4009428c0e8` |
| ComfyUI compatibility fork | [endman100/YuE](https://github.com/endman100/YuE) | [`dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349`](https://github.com/endman100/YuE/commit/dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349) |
| ComfyUI-YuE2 parity implementation | [endman100/ComfyUI-YuE2](https://github.com/endman100/ComfyUI-YuE2) | [`08c0e9ba93aba65b87fdeb985dbc28cc1c848e8c`](https://github.com/endman100/ComfyUI-YuE2/commit/08c0e9ba93aba65b87fdeb985dbc28cc1c848e8c) |

The compatibility fork is packaged as `yue2-infer-comfyui==0.1.6`. It keeps the installed ComfyUI Torch environment authoritative, uses ComfyUI's selected optimized attention backend, and avoids the upstream CUDA Graph allocator conflict. It does not replace ComfyUI.

## Integration boundaries

| Official stage | Integration | ComfyUI contract |
| --- | --- | --- |
| `YuE2Pipeline.from_pretrained()` | **YuE2 Model Loader** | `MODEL` plus private lazy runtime |
| `pipe.plan()` | **YuE2 Plan Score / Plan Editor** | `YUE2_PLAN`, `STRING`, `DICT` |
| `pipe.generate_semantic()` | **YuE2 Generate Semantic** | `CONDITIONING` |
| Allocate acoustic state | **YuE2 Empty Latent** | `LATENT` |
| Seeded acoustic noise | Native **RandomNoise** | `NOISE` |
| Flow guidance | Native **BasicGuider** | `GUIDER` |
| Explicit-midpoint solver | **YuE2 Explicit Midpoint Sampler** | `SAMPLER` |
| Linear `1 → 0` grid | **YuE2 Linear Schedule** | `SIGMAS` |
| Execute flow | Native **SamplerCustomAdvanced** | `LATENT` |
| `pipe.decode()` | **YuE2 Decode** | `AUDIO` |
| Listen or save | Native **Preview Audio / Save Audio** | `AUDIO` |

YuE2 is flow matching, but the official synthesis path also specifies semantic objects, chunk boundaries, token-major seeded noise, the complete `1 → 0` grid, and chunk-major explicit-midpoint integration. The custom `SAMPLER` and `SIGMAS` providers preserve those rules while native `SamplerCustomAdvanced` owns execution, callbacks, preview, and cancellation.

## Comparison method

[`tests/integration_parity.py`](../tests/integration_parity.py) generates one **YuE2 Generate Semantic** result, then feeds the same semantic object to:

1. Official `YuE2Pipeline.synthesize()` followed by `decode()`.
2. ComfyUI `RandomNoise` → `BasicGuider` → YuE2 `SAMPLER`/`SIGMAS` → `SamplerCustomAdvanced` → **YuE2 Decode**.

Sharing semantic tokens isolates the acoustic flow graph and decoder. Independently sampling the autoregressive semantic stage would primarily measure CUDA RNG repeatability; plan and semantic contracts are covered separately by the unit suite.

The acceptance rule was defined as exact tensor equality at the latent and raw float32 audio boundaries: equal shapes, `numpy.array_equal == true`, maximum absolute error `0`, and RMSE `0`. Matched WAVs were saved as PCM-24 and required identical SHA-256 hashes.

## Environment

| Item | Tested value |
| --- | --- |
| Date | 2026-09-15 |
| ComfyUI | `0.35.0`, commit `40c4fcdf513a4523e39d54a9d391908af8df8171` |
| V3 API / frontend | `comfy_api.v0_0_2` / `1.51.10` |
| Runtime fork | `0.1.6`, commit `dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349` |
| Python / PyTorch / CUDA | Python 3.12.8 / PyTorch 2.14.0+cu130 / CUDA 13.0 |
| Main dependencies | Transformers 4.57.6, Hugging Face Hub 0.36.2, Accelerate 1.12.0, Safetensors 0.8.0, SoundFile 0.14.0, tiktoken 0.12.0 |
| Hardware / precision | NVIDIA GeForce RTX 5090 / BF16 main and NAR, FP32 VAE comparison |
| Solver | Explicit midpoint, 32 steps, linear `1 → 0` schedule |

## Results

| Case | Coverage | Tokens | Decode | Duration | Latent max abs / RMSE | Audio max abs / RMSE | Result |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| `01_pop_vocal_seed_123` | Vocal pop, `cot=off`, seed 123 | 200 | Tiled | 7.9987 s | `0 / 0` | `0 / 0` | Pass |
| `02_electronic_instrumental_seed_2026` | Instrumental electronic, custom semantic sampling | 160 | Tiled | 6.3987 s | `0 / 0` | `0 / 0` | Pass |
| `03_rock_vocal_seed_987654` | Vocal rock, high-temperature semantic sampling | 128 | Tiled | 5.1187 s | `0 / 0` | `0 / 0` | Pass |
| `04_auto_abc_full_seed_42` | Automatically generated `full` ABC score | 128 | Tiled | 5.1187 s | `0 / 0` | `0 / 0` | Pass |
| `05_supplied_abc_melody_seed_314159` | Supplied ABC rearrangement, `cot=melody` | 96 | Full | 3.8387 s | `0 / 0` | `0 / 0` | Pass |

Result: **5/5 passed**. The complete run took 103.80 seconds including initial loading.

## Artifacts

| Case | Official | ComfyUI | Difference | Report | Matched WAV SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `01_pop_vocal_seed_123` | [WAV](../validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/official.wav) | [WAV](../validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/comfyui.wav) | [WAV](../validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/difference.wav) | [JSON](../validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/report.json) | `6e61a84f18bc7de40ee440270cb283685caba752bbcd58044ec031cce685bc00` |
| `02_electronic_instrumental_seed_2026` | [WAV](../validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/official.wav) | [WAV](../validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/comfyui.wav) | [WAV](../validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/difference.wav) | [JSON](../validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/report.json) | `c74e475ce3fa095c4fea142011d474411de4cf448cc6af2495de6c661efca05a` |
| `03_rock_vocal_seed_987654` | [WAV](../validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/official.wav) | [WAV](../validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/comfyui.wav) | [WAV](../validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/difference.wav) | [JSON](../validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/report.json) | `da6a4503f97b74303616a7e6bdbd7ed8cec619077d7a8e86cb2daf70d738226b` |
| `04_auto_abc_full_seed_42` | [WAV](../validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/official.wav) | [WAV](../validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/comfyui.wav) | [WAV](../validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/difference.wav) | [JSON](../validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/report.json) | `8593756d3d9441ecc5e8a550f0e2acc92c8bcfc48d6f05527060b33fd33be8a1` |
| `05_supplied_abc_melody_seed_314159` | [WAV](../validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/official.wav) | [WAV](../validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/comfyui.wav) | [WAV](../validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/difference.wav) | [JSON](../validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/report.json) | `9cac2dd1b0ef47a4fa189504d9e14bd22dbb1ed93c0babb828a4f447c4c93543` |

The aggregate report is [`summary.json`](../validation_outputs/yue2_parity_5cases/summary.json). The MP4 players in the project README are delivery previews encoded from the ComfyUI WAVs; they are not comparison inputs.

## Reproduce

From the repository root in the configured ComfyUI Python environment:

```powershell
python -m pytest -q
python tests/integration_parity.py
```

The recorded local unit run passes 26 tests with two third-party SWIG deprecation warnings. The integration run requires the pinned model snapshots and a supported CUDA device. Newly generated artifacts are written to `ComfyUI/output/yue2_parity_5cases/<case>/`.
