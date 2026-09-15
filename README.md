# ComfyUI-YuE2

Composable ComfyUI V3 nodes for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), a lyrics-and-style-to-song model with editable symbolic score planning. The package keeps YuE2-specific planning, autoregressive semantic generation, flow sampling, and audio decoding behind eight nodes while reusing ComfyUI's native `MODEL`, `CONDITIONING`, `LATENT`, `SAMPLER`, `SIGMAS`, `AUDIO`, guider, noise, sampling executor, preview, and save contracts.

## Original project and tested revisions

This repository is a ComfyUI integration, not a replacement for the official YuE2 project.

| Component | Source | Revision used by this integration |
| --- | --- | --- |
| Official YuE2 source | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | [`0edaf2f4053ef4731334b8329834b107977f9637`](https://github.com/multimodal-art-projection/YuE/commit/0edaf2f4053ef4731334b8329834b107977f9637) |
| YuE2-3B weights | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | [`1a96eca688d6ae5d7f0feb88573fec89920fcd19`](https://huggingface.co/m-a-p/YuE2-3B/commit/1a96eca688d6ae5d7f0feb88573fec89920fcd19) |
| YuE2 VAE weights | [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | `95535e72a97bc0f09b8ada125d26b4009428c0e8` |
| ComfyUI compatibility fork | [endman100/YuE](https://github.com/endman100/YuE) | [`dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349`](https://github.com/endman100/YuE/commit/dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349), based on the official revision above |
| ComfyUI-YuE2 parity implementation | [endman100/ComfyUI-YuE2](https://github.com/endman100/ComfyUI-YuE2) | [`08c0e9ba93aba65b87fdeb985dbc28cc1c848e8c`](https://github.com/endman100/ComfyUI-YuE2/commit/08c0e9ba93aba65b87fdeb985dbc28cc1c848e8c) |

The compatibility fork is installed as `yue2-infer-comfyui==0.1.6`. Its narrow purpose is to keep the current ComfyUI Torch environment authoritative, route YuE2 attention through ComfyUI's selected optimized backend, and avoid the upstream CUDA Graph allocator conflict. The model architecture, tokenization, planning, sampling equations, and decoder behavior remain YuE2's.

The loader's revision fields are optional. A blank revision follows the Hugging Face repository default; enter the full tested hashes above when a reproducible, immutable model snapshot is required.

## Project goals and features

| Capability | Status | Implementation |
| --- | --- | --- |
| Lyrics and style to complete song | Supported | Official YuE2 plan, semantic, acoustic flow, and VAE stages |
| Symbolic planning | Supported | `full` melody-and-chord planning, `melody` planning, or `off` |
| Supplied or edited score | Supported | Human-readable ABC notation through YuE2 Plan Score and Plan Editor |
| Cover and rearrangement | Supported from an ABC score | Supply a separately transcribed or hand-written score; no transcription model is bundled |
| Native ComfyUI sampling composition | Supported | `RandomNoise` + `BasicGuider` + `SamplerCustomAdvanced` with custom `SAMPLER` and `SIGMAS` providers |
| Standard graph types | Supported | `MODEL`, `CONDITIONING`, `LATENT`, `SAMPLER`, `SIGMAS`, `AUDIO`, and `DICT` |
| Lazy loading and offload | Supported | The loader creates a lightweight handle; models load only when a downstream stage executes and use ComfyUI model patchers |
| Automatic model download | Supported | User-triggered first execution downloads an allowlisted Hugging Face snapshot into the YuE2 cache |
| Local and extra model paths | Supported | `models/yue2`, absolute local directories, and `extra_model_paths.yaml` roots |
| Precision selection | Supported | `auto`, BF16, FP16, FP32, and NVIDIA FP8 AR weights; VAE remains FP32 |
| Full and tiled decode | Supported | 48 kHz stereo `AUDIO`; tiled decode lowers peak memory |
| Progress and cancellation | Supported for score tokens, semantic tokens, sampling, and tiled decode | Uses ComfyUI progress and interruption checks |
| Multiple songs as one tensor batch | Not exposed | One song is generated per workflow execution; ComfyUI queueing can run multiple prompts |
| Waveform reference conditioning, audio inpaint, audio continuation, or stem editing | Not implemented | These are not conditioning paths exposed by the integrated YuE2 inference API |
| Demucs, Seed-VC, SheetSage2, or local LLM tools | Not bundled | These require separate models or methods and are intentionally outside this node pack |

YuE2 editing is score-level regeneration: change the ABC score, style, or lyrics and render a new complete recording. It is not waveform inpainting. For a cover, obtain ABC notation with an external transcription tool, review it, and then use `cot=melody` or `cot=full` with the supplied score.

## Integration approach

| Official YuE2 stage | Custom node or adapter | Native ComfyUI contract | Why this boundary exists |
| --- | --- | --- | --- |
| `YuE2Pipeline.from_pretrained()` | **YuE2 Model Loader** | `MODEL` plus shared `YUE2_RUNTIME` | Exposes a flow model to ComfyUI while retaining model-specific tokenizer and decoder state in a lazy runtime |
| `pipe.plan()` | **YuE2 Plan Score** | Custom `YUE2_PLAN`, plus optional text preview | Score planning is autoregressive and carries YuE2-specific structured state |
| Edit and rebuild a plan | **YuE2 Plan Editor** | `YUE2_PLAN`, `STRING`, `DICT` | Lets users change safe public fields while rebuilding tokenizer-derived fields |
| `pipe.generate_semantic()` | **YuE2 Generate Semantic** | `CONDITIONING` | Carries the official semantic-token result into `BasicGuider` without exposing raw model-private sockets |
| Allocate acoustic state | **YuE2 Empty Latent** | `LATENT` | Derives the official chunk layout and 64-channel length from semantic conditioning |
| Seeded acoustic noise | ComfyUI **RandomNoise** | `NOISE` | Reuses the core seed provider; the sampler remaps layout to YuE2's token-major RNG order |
| Flow guidance | ComfyUI **BasicGuider** | `GUIDER` | Uses the native custom-sampling executor contract |
| `pipe.synthesize()` explicit midpoint solver | **YuE2 Explicit Midpoint Sampler** | `SAMPLER` | Preserves YuE2's chunk-major RK2 update order, which a generic solver cannot infer |
| Official `1 → 0` flow grid | **YuE2 Linear Schedule** | `SIGMAS` | Supplies the exact YuE2 schedule, including both endpoints |
| Execute the acoustic flow | ComfyUI **SamplerCustomAdvanced** | `LATENT` | Reuses ComfyUI's noise, guider, callback, preview, and cancellation plumbing |
| `pipe.decode()` | **YuE2 Decode** | `AUDIO` | Converts YuE2's 64-channel acoustic latent to 48 kHz stereo audio |
| Preview or save | ComfyUI **Preview Audio / Save Audio** | `AUDIO` | Avoids a redundant custom output node |

YuE2 is a flow-matching model, but the official synthesis contract is more specific than “any flow model”: it uses semantic objects, official chunk boundaries, a token-major seeded noise draw, a linear `1 → 0` grid, and chunk-major explicit-midpoint integration. This package therefore uses `SamplerCustomAdvanced` with a small YuE2 `SAMPLER` and `SIGMAS` provider instead of duplicating the whole sampling executor or claiming that an arbitrary `KSampler` configuration is equivalent.

## Installation and model setup

### Requirements

- ComfyUI `>=0.35.0`
- Python `>=3.10`; tested with Python 3.12.8
- A supported ComfyUI device. The primary tested path is NVIDIA CUDA with BF16 support.
- Approximately 7.8 GB of model files for the tested YuE2-3B and default VAE snapshots, plus runtime memory and cache space.

### ComfyUI Manager / Registry

The Registry ID is [`yue2-native`](https://registry.comfy.org/nodes/yue2-native). The node entry is active, but on 2026-09-15 the Registry API reports releases `0.1.0` and `0.1.1` as `Flagged`. Until an installable version is visible in Manager, use the manual installation below. When the status is cleared, the intended CLI command is:

```powershell
comfy node install yue2-native
```

### Manual installation

Use the Python environment that launches ComfyUI:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/endman100/ComfyUI-YuE2.git
cd ComfyUI-YuE2
python -m pip install -r requirements.txt
```

Restart ComfyUI after installation. The dependency is commit-pinned to the compatibility fork; it does not install or replace ComfyUI itself.

### Automatic download

With **Weight Source** set to `hugging_face`, the default repositories are:

- Main model: `m-a-p/YuE2-3B`
- VAE: `m-a-p/YuE2-Vae`

Creating the loader node does not download or load weights. The first downstream YuE2 stage resolves the two snapshots through `huggingface_hub.snapshot_download`. The download is restricted to the model configuration, tokenizer, manifest, model code, documented notices, and Safetensors weight files. The resolved model manifest is verified when **Verify Downloaded Hashes** is enabled. Hugging Face cache transfer and partial-file handling are used; a failed transfer is not treated as a complete snapshot.

The default cache is:

```text
ComfyUI/models/yue2/hub/
```

Set **Offline / Local Files Only** after a complete snapshot is cached. If access authentication is ever required, configure Hugging Face credentials outside the workflow; do not place tokens in workflow JSON.

### Local models

Set **Weight Source** to `local`. **Model ID / Path** and **VAE ID / Path** may be absolute directories or relative directories below a registered `yue2` model root. A typical layout is:

```text
ComfyUI/
└── models/
    └── yue2/
        ├── YuE2-3B/
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── qwen.tiktoken
        │   └── weights_manifest.json
        └── YuE2-Vae/
            ├── config.json
            ├── model.safetensors
            └── weights_manifest.json
```

Use `YuE2-3B` and `YuE2-Vae` as the two relative path inputs for that layout. Paths are resolved and checked again when the pipeline opens. Local configuration, tokenizer, manifests, weight sizes, and modification timestamps participate in the loader fingerprint.

### `extra_model_paths.yaml`

The package registers the `yue2` model category with ComfyUI. For a shared model directory, add an entry like this to `extra_model_paths.yaml` and restart ComfyUI:

```yaml
shared_models:
  base_path: D:/AI/models
  yue2: yue2
```

The example above discovers models below `D:/AI/models/yue2/`. Relative local paths must remain inside a configured `yue2` root. **Cache Directory** may separately override the Hugging Face cache used by `hugging_face` mode.

### Precision and memory behavior

| Loader setting | Main AR / score / semantic stages | NAR acoustic stage | VAE | Hardware notes |
| --- | --- | --- | --- | --- |
| `auto` | BF16 on supported CUDA, FP16 on other CUDA or MPS, FP32 on CPU | Same selected compute dtype | FP32 | Recommended default |
| `bfloat16` | BF16 | BF16 | FP32 | CUDA device must support BF16 |
| `float16` | FP16 | FP16 | FP32 | Useful where BF16 is unavailable |
| `float32` | FP32 | FP32 | FP32 | Highest memory use; CPU-compatible but slow |
| `fp8` | BF16 compute with official FP8-quantized AR linear weights | Exact BF16 weights restored | FP32 | NVIDIA CUDA compute capability 8.9 or newer |

The main model and VAE are wrapped in ComfyUI `CoreModelPatcher` objects. **Offload AR Stage** unloads the main language model before VAE loading to reduce peak VRAM. Disable **Full Decode** to use bounded tiled decode with overlap; enable it only when the official full-sequence decode path fits in memory.

## How to use

1. Open **Workflow Templates → Custom Nodes → ComfyUI-YuE2** and load `yue2_text_to_song_bf16`.
2. In **YuE2 Model Loader**, choose the model source, model/VAE locations, dtype, and device.
3. In **YuE2 Plan Score**, edit **Style Prompt** and **Lyrics**. Choose `full`, `melody`, or `off` under **Score Planning**.
4. Use the same numeric seed in **Plan Seed** and ComfyUI's **RandomNoise** when you want the template's fully repeatable seeded path.
5. Set **Maximum Semantic Tokens** to control the generation work and approximate duration. Keep the shipped sampling defaults for the first run.
6. Queue the workflow. The terminal `Save Audio` node writes FLAC under `ComfyUI/output/audio/`; connect the `AUDIO` output to any compatible native preview, save, or processing node.

### ABC score and `YUE2_PLAN`

ABC is human-readable text notation for melody, rhythm, meter, key, tempo, and optional chord symbols. In **YuE2 Plan Score**:

- `full` with a blank ABC field generates melody and chord notation.
- `melody` with a blank ABC field generates a melody plan.
- `full` or `melody` with supplied ABC uses that score as musical structure.
- `off` skips symbolic score planning.

`YUE2_PLAN` is a structured Python `SymbolicPlan`, not a JSON string. It contains the request, ABC, tokenized score, prefix tokens, timing, and truncation state. The separate **ABC Score (text preview/export)** output is not required for generation because the plan already contains the score; connect it only to inspect, copy, or save the text.

### Plan Editor

**YuE2 Plan Editor** supports two paths:

- Put replacement notation in **ABC Score Override** for the common score-only edit.
- Put a JSON object in **Advanced Overrides** for `style`, `lyrics`, `cot`, `seed`, `cfg_scale`, `abc`, or `id`, for example `{"style":"jazz ballad","seed":42}`.

Blank override fields keep the source plan values. The node ignores supplied derived fields such as `abc_ids`, `prefix`, `timing`, and `truncated`, then rebuilds them through the official tokenizer and planning API. **Plan Data (DICT)** is provided for generic ComfyUI dictionary inspection or conversion; feed **Edited Song Plan** into **YuE2 Generate Semantic**.

## Nodes

| Node | Purpose | Main inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| **YuE2 Model Loader** | Creates the lazy model/runtime pair | source, model, VAE, precision, device, memory/offload, revisions, cache | `MODEL`, `YUE2_RUNTIME` | Downloads and loads only after a downstream YuE2 stage runs; implements deterministic local-file fingerprinting |
| **YuE2 Plan Score** | Creates a plan from style, lyrics, and optional ABC | runtime, style, lyrics, CoT mode, plan seed, score sampling | `YUE2_PLAN`, ABC `STRING` | Uses the official autoregressive YuE2 model for score planning |
| **YuE2 Plan Editor** | Safely edits an existing plan | runtime, source plan, ABC override, JSON overrides | `YUE2_PLAN`, ABC `STRING`, `DICT` | Rebuilds all tokenizer-derived fields |
| **YuE2 Generate Semantic** | Generates discrete semantic music tokens | runtime, plan, semantic sampling | `CONDITIONING` | Uses the official trained YuE2 autoregressive language model; connect to both BasicGuider and Empty Latent |
| **YuE2 Empty Latent** | Allocates the official acoustic latent chunks | YuE2 `CONDITIONING` | `LATENT` | Reads semantic-token length and creates a nested 64-channel latent |
| **YuE2 Explicit Midpoint Sampler** | Provides YuE2's chunk-major RK2 solver | none | `SAMPLER` | Connect to `SamplerCustomAdvanced` |
| **YuE2 Linear Schedule** | Provides the official flow time grid | steps | `SIGMAS` | Exact `1 → 0` linear grid; 32 steps by default |
| **YuE2 Decode** | Decodes sampled acoustic latents | runtime, sampled latent, full/tiled mode | `AUDIO` | Returns 48 kHz stereo without peak/RMS/loudness normalization; only finite checking and `[-1,1]` sample clamping are applied |

There is intentionally no custom saver, preview node, VAE encode node, one-shot generator, or sampling-config wrapper. Use ComfyUI's native `SamplerCustomAdvanced`, audio preview, and audio save nodes. VAE encode would only provide a codec round trip: the integrated YuE2 API does not accept encoded audio latents as reference conditioning for generation.

## Example workflows

ComfyUI discovers the JSON files under `example_workflows` and lists them under **Workflow Templates → Custom Nodes → ComfyUI-YuE2**.

| Workflow | Purpose | Required models | Edit these inputs | Expected terminal output |
| --- | --- | --- | --- | --- |
| [`yue2_text_to_song_bf16.json`](example_workflows/yue2_text_to_song_bf16.json) | Recommended minimal text-to-song graph with direct (`cot=off`) generation | YuE2-3B + YuE2-Vae | Style, lyrics, Plan/RandomNoise seed, semantic token limits | FLAC under `output/audio/yue2_text_to_song*` |
| [`yue2_automatic_score.json`](example_workflows/yue2_automatic_score.json) | Generates a `full` ABC melody-and-chord plan before rendering | YuE2-3B + YuE2-Vae | Style, lyrics, plan seed, score/semantic sampling | FLAC under `output/audio/yue2_automatic_score*` plus ABC preview |
| [`yue2_abc_cover_rearrangement.json`](example_workflows/yue2_abc_cover_rearrangement.json) | Supplies and optionally edits ABC for a cover or rearrangement | YuE2-3B + YuE2-Vae; source ABC must already exist | ABC, CoT mode, style, lyrics, Plan Editor overrides, both seeds | FLAC under `output/audio/yue2_abc_rearrangement*` plus edited Plan Data/ABC preview |
| [`yue2_text_to_song_fp8_nvidia.json`](example_workflows/yue2_text_to_song_fp8_nvidia.json) | Lower-memory FP8 AR text-to-song path | YuE2-3B + YuE2-Vae; NVIDIA compute capability 8.9+ | Style, lyrics, both seeds, VRAM budget | FLAC under `output/audio/yue2_fp8_text_to_song*` |

All four templates use only this package's eight nodes plus current ComfyUI core nodes. They do not require Demucs, Seed-VC, SheetSage2, or a local LLM.

## Implementation results versus official results

The matched parity test is implemented in [`tests/integration_parity.py`](tests/integration_parity.py). It first produces one **YuE2 Generate Semantic** result, then gives that exact semantic object to two branches:

1. Official `YuE2Pipeline.synthesize()` followed by `decode()`.
2. ComfyUI `RandomNoise` → `BasicGuider` → YuE2 `SAMPLER`/`SIGMAS` → `SamplerCustomAdvanced` → **YuE2 Decode**.

Sharing semantic tokens is intentional. It isolates whether ComfyUI's acoustic flow graph and decoder reproduce the official `synthesize/decode` implementation; independently sampling the autoregressive semantic stage would primarily measure CUDA RNG repeatability. Plan and semantic node contracts are covered separately by the unit suite.

### Matched environment

| Item | Tested value |
| --- | --- |
| Date | 2026-09-15 |
| ComfyUI | `0.35.0`, commit `40c4fcdf513a4523e39d54a9d391908af8df8171` |
| V3 API | `comfy_api.v0_0_2` |
| Frontend | `1.51.10` |
| ComfyUI-YuE2 implementation | `08c0e9ba93aba65b87fdeb985dbc28cc1c848e8c` |
| YuE2 runtime fork | `0.1.6`, commit `dc90522a2f6c94cdfce0d0db72a3c6ee54a2a349` |
| Main model | `m-a-p/YuE2-3B@1a96eca688d6ae5d7f0feb88573fec89920fcd19` |
| VAE | `m-a-p/YuE2-Vae@95535e72a97bc0f09b8ada125d26b4009428c0e8` |
| Python / PyTorch / CUDA | Python 3.12.8 / PyTorch 2.14.0+cu130 / CUDA 13.0 |
| Main dependencies | Transformers 4.57.6, Hugging Face Hub 0.36.2, Accelerate 1.12.0, Safetensors 0.8.0, SoundFile 0.14.0, tiktoken 0.12.0 |
| Hardware | NVIDIA GeForce RTX 5090 |
| Precision | BF16 main model and NAR, FP32 VAE/audio comparison |
| Solver | Explicit midpoint, 32 steps, linear `1 → 0` schedule |

The newest stable ComfyUI release at the test date was 0.35.0. The parity run used frontend 1.51.10; this package has no custom frontend JavaScript.

### Acceptance rule and results

The acceptance threshold was exact tensor equality at both comparison boundaries: identical shape, `numpy.array_equal == true`, maximum absolute error `0`, and RMSE `0`. WAV files were saved from the matched float32 arrays as PCM-24; each official/ComfyUI pair also has the same SHA-256.

| Case | Coverage | Semantic tokens | Decode | Duration | Latent max abs / RMSE | Audio max abs / RMSE | Result | Official = ComfyUI WAV SHA-256 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| `01_pop_vocal_seed_123` | Vocal pop, `cot=off`, seed 123 | 200 | Tiled | 7.9987 s | `0 / 0` | `0 / 0` | Pass | `6e61a84f18bc7de40ee440270cb283685caba752bbcd58044ec031cce685bc00` |
| `02_electronic_instrumental_seed_2026` | Instrumental electronic, custom semantic sampling | 160 | Tiled | 6.3987 s | `0 / 0` | `0 / 0` | Pass | `c74e475ce3fa095c4fea142011d474411de4cf448cc6af2495de6c661efca05a` |
| `03_rock_vocal_seed_987654` | Vocal rock, high-temperature semantic sampling | 128 | Tiled | 5.1187 s | `0 / 0` | `0 / 0` | Pass | `da6a4503f97b74303616a7e6bdbd7ed8cec619077d7a8e86cb2daf70d738226b` |
| `04_auto_abc_full_seed_42` | Automatically generated `full` ABC score | 128 | Tiled | 5.1187 s | `0 / 0` | `0 / 0` | Pass | `8593756d3d9441ecc5e8a550f0e2acc92c8bcfc48d6f05527060b33fd33be8a1` |
| `05_supplied_abc_melody_seed_314159` | Supplied ABC rearrangement, `cot=melody` | 96 | Full | 3.8387 s | `0 / 0` | `0 / 0` | Pass | `9cac2dd1b0ef47a4fa189504d9e14bd22dbb1ed93c0babb828a4f447c4c93543` |

Result: **5/5 passed**, with exact latent and raw float32 audio equality. The run completed in 103.80 seconds including initial loading. Per-case `official.wav`, `comfyui.wav`, `difference.wav`, and `report.json` files are generated locally under:

```text
ComfyUI/output/yue2_parity_5cases/<case>/
```

Reproduce the checks from the repository root inside the configured ComfyUI Python environment:

```powershell
python -m pytest -q
python tests/integration_parity.py
```

The current local unit run passes **26 tests**. Two third-party SWIG deprecation warnings are emitted; there are no test failures. The integration test requires the pinned model snapshots to be cached and a supported CUDA device.

## Troubleshooting

- **Out of memory:** keep **Offload AR Stage** enabled, use tiled decode, close other GPU workloads, or use the FP8 workflow on supported NVIDIA hardware. Reducing semantic token limits shortens the requested generation.
- **Missing files while offline:** disable **Offline / Local Files Only** for the first download, or verify both complete snapshots exist in the selected cache.
- **Hash verification failure:** remove only the reported corrupt snapshot/file and let Hugging Face download it again. Do not disable verification to hide a damaged weight.
- **Conditioning/runtime mismatch:** connect `MODEL`, `YUE2_RUNTIME`, and `CONDITIONING` from the same **YuE2 Model Loader** chain.
- **Wrong latent or generic sampler error:** use **YuE2 Empty Latent**, **YuE2 Explicit Midpoint Sampler**, **YuE2 Linear Schedule**, and `SamplerCustomAdvanced` together.
- **Score edit rejected:** `full` and `melody` plans require non-empty ABC after editing; use `cot=off` only when intentionally removing the score.
- **Need a reference recording or waveform edit:** first convert the musical content to reviewed ABC with a separate tool. This node pack does not accept source audio as YuE2 conditioning.

## License

The original ComfyUI integration code in this repository is released under the MIT License. YuE2 model weights are separate downloads licensed under CC BY-NC 4.0 and are not included in this repository. The YuE2 inference dependency and its bundled components retain their upstream terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or commercial use.
