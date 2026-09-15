# ComfyUI-YuE2

Composable ComfyUI V3 nodes for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B): generate songs from lyrics and style, create or edit an ABC score, and run YuE2's acoustic flow through native ComfyUI sampling nodes.

[Original YuE2](https://github.com/multimodal-art-projection/YuE) · [Example workflows](example_workflows) · [Model setup](docs/MODEL_SETUP.md) · [Full parity report](docs/VALIDATION.md)

## Highlights

- Eight focused custom nodes; native `RandomNoise`, `BasicGuider`, `SamplerCustomAdvanced`, `Preview Audio`, and `Save Audio` handle the standard graph work.
- Native `MODEL`, `CONDITIONING`, `LATENT`, `SAMPLER`, `SIGMAS`, `AUDIO`, and `DICT` connections.
- Lazy loading, ComfyUI model patchers, AR-stage offload, progress reporting, and cancellation checks.
- User-triggered automatic download from fixed model repositories, plus local paths and `extra_model_paths.yaml`.
- `auto`, BF16, FP16, FP32, and NVIDIA FP8 AR weight modes. The VAE remains FP32.
- Score planning modes `full`, `melody`, and `off`, with an editor for supplied or generated ABC notation.

YuE2 editing is score-level regeneration. This integration does not claim waveform inpainting, audio-reference conditioning, continuation, source separation, or voice conversion.

## Listen to generated samples

Expand a case, press play, and unmute the speaker once if Chrome starts muted. Each player contains the **ComfyUI graph output**, transcoded from WAV to AAC-in-MP4 only so GitHub renders inline playback. In all five matched cases, the official pipeline WAV and ComfyUI WAV are sample-identical; parity metrics use the original 48 kHz stereo WAV arrays, not the listening transcode.

<details open>
<summary><strong>▶ 01 · Pop vocal · cot=off · seed 123</strong></summary>

https://github.com/user-attachments/assets/0f4e96ba-9da5-406c-a2f6-6693086bdbab

[Official WAV](validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/official.wav) · [ComfyUI WAV](validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/comfyui.wav) · [Difference](validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/difference.wav) · [Metrics](validation_outputs/yue2_parity_5cases/01_pop_vocal_seed_123/report.json)

</details>

<details>
<summary><strong>▶ 02 · Electronic instrumental · seed 2026</strong></summary>

https://github.com/user-attachments/assets/53aa49fb-9e69-402e-bffc-84991615f02d

[Official WAV](validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/official.wav) · [ComfyUI WAV](validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/comfyui.wav) · [Difference](validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/difference.wav) · [Metrics](validation_outputs/yue2_parity_5cases/02_electronic_instrumental_seed_2026/report.json)

</details>

<details>
<summary><strong>▶ 03 · Rock vocal · high-temperature semantic sampling · seed 987654</strong></summary>

https://github.com/user-attachments/assets/b6bd61dc-2c91-45de-9089-0bdd89cd3ac9

[Official WAV](validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/official.wav) · [ComfyUI WAV](validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/comfyui.wav) · [Difference](validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/difference.wav) · [Metrics](validation_outputs/yue2_parity_5cases/03_rock_vocal_seed_987654/report.json)

</details>

<details>
<summary><strong>▶ 04 · Automatically planned full ABC score · seed 42</strong></summary>

https://github.com/user-attachments/assets/be9e411c-2714-4eba-8d24-8b98f3fb1d89

[Official WAV](validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/official.wav) · [ComfyUI WAV](validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/comfyui.wav) · [Difference](validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/difference.wav) · [Metrics](validation_outputs/yue2_parity_5cases/04_auto_abc_full_seed_42/report.json)

</details>

<details>
<summary><strong>▶ 05 · Supplied ABC melody rearrangement · seed 314159</strong></summary>

https://github.com/user-attachments/assets/0f2b2003-18bf-4d72-9816-aa50dd07395b

[Official WAV](validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/official.wav) · [ComfyUI WAV](validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/comfyui.wav) · [Difference](validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/difference.wav) · [Metrics](validation_outputs/yue2_parity_5cases/05_supplied_abc_melody_seed_314159/report.json)

</details>

## Install

Requirements: ComfyUI `>=0.35.0`, Python `>=3.10`, and approximately 7.8 GB for the tested YuE2-3B and VAE snapshots. The primary tested path is NVIDIA CUDA with BF16 support.

The Registry ID is [`yue2-native`](https://registry.comfy.org/nodes/yue2-native). Releases `0.1.0` and `0.1.1` are currently flagged by the Registry, so use the manual install until an installable version appears in Manager:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/endman100/ComfyUI-YuE2.git
cd ComfyUI-YuE2
python -m pip install -r requirements.txt
```

Restart ComfyUI after installation. See [Model setup](docs/MODEL_SETUP.md) for automatic download, local folders, `extra_model_paths.yaml`, dtype, and memory behavior.

## Quick start

1. Open **Workflow Templates → Custom Nodes → ComfyUI-YuE2**.
2. Load `yue2_text_to_song_bf16`.
3. Edit **Style Prompt** and **Lyrics** in **YuE2 Plan Score**.
4. Keep the shipped sampling defaults, queue the graph, and listen through the native **Preview Audio** or **Save Audio** node.

Use the same numeric value in **Plan Seed** and native **RandomNoise** when you want the template's repeatable seeded path. **Maximum Semantic Tokens** controls generation work and approximate duration.

## Example workflows

| Workflow | Use it for | Terminal output |
| --- | --- | --- |
| [`yue2_text_to_song_bf16.json`](example_workflows/yue2_text_to_song_bf16.json) | Recommended text-to-song path with `cot=off` | `output/audio/yue2_text_to_song*` |
| [`yue2_automatic_score.json`](example_workflows/yue2_automatic_score.json) | Generate a full melody-and-chord ABC plan before rendering | Audio plus ABC preview |
| [`yue2_abc_cover_rearrangement.json`](example_workflows/yue2_abc_cover_rearrangement.json) | Supply and edit an existing ABC score for a cover or rearrangement | Audio plus edited plan data and ABC preview |
| [`yue2_text_to_song_fp8_nvidia.json`](example_workflows/yue2_text_to_song_fp8_nvidia.json) | Lower-memory FP8 AR path on NVIDIA compute capability 8.9+ | `output/audio/yue2_fp8_text_to_song*` |

All four templates use only this package's eight nodes plus current ComfyUI core nodes. No Demucs, Seed-VC, SheetSage2, or local LLM is required.

## Nodes

| Node | Purpose | Output / native composition |
| --- | --- | --- |
| **YuE2 Model Loader** | Create a lazy model/runtime pair and resolve automatic or local weights | `MODEL`, `YUE2_RUNTIME` |
| **YuE2 Plan Score** | Build a song plan from style, lyrics, and optional ABC | `YUE2_PLAN`, ABC `STRING` |
| **YuE2 Plan Editor** | Safely replace ABC or public plan fields and rebuild derived tokens | `YUE2_PLAN`, ABC `STRING`, `DICT` |
| **YuE2 Generate Semantic** | Generate semantic music tokens with the official trained AR model | `CONDITIONING` → `BasicGuider` and **YuE2 Empty Latent** |
| **YuE2 Empty Latent** | Allocate the official 64-channel acoustic latent layout | `LATENT` → `SamplerCustomAdvanced` |
| **YuE2 Explicit Midpoint Sampler** | Provide YuE2's chunk-major RK2 update rule | `SAMPLER` → `SamplerCustomAdvanced` |
| **YuE2 Linear Schedule** | Provide the exact `1 → 0` flow grid | `SIGMAS` → `SamplerCustomAdvanced` |
| **YuE2 Decode** | Decode sampled latents without loudness normalization | 48 kHz stereo `AUDIO` → native preview/save nodes |

There is intentionally no custom saver, preview, VAE encode, one-shot generator, or sampling wrapper. Existing ComfyUI nodes already own those compatible responsibilities.

## ABC score and Plan Editor

ABC is editable text notation for melody, rhythm, meter, key, tempo, and optional chord symbols.

- `full` with blank ABC generates melody and chord notation.
- `melody` with blank ABC generates a melody plan.
- `full` or `melody` with supplied ABC follows that score as musical structure.
- `off` skips symbolic planning.

`YUE2_PLAN` is a structured Python object, not a JSON string. The ABC text output is for preview or export; generation already receives it inside the plan. In **YuE2 Plan Editor**, use **ABC Score Override** for a normal score edit or **Advanced Overrides** for public JSON fields such as `{"style":"jazz ballad","seed":42}`. Derived token fields are rebuilt automatically.

## Official parity result

Five matched BF16/CUDA cases passed with exact latent and raw float32 audio equality: `numpy.array_equal == true`, maximum absolute error `0`, RMSE `0`, and identical official/ComfyUI WAV SHA-256 hashes. The run used ComfyUI 0.35.0, frontend 1.51.10, Python 3.12.8, PyTorch 2.14.0+cu130, CUDA 13.0, and an RTX 5090.

Read the [full reproducible parity report](docs/VALIDATION.md) for pinned revisions, method, per-case metrics, hashes, and download links. Validation WAVs remain in the repository but are excluded from the Comfy Registry package through `.comfyignore`.

## Troubleshooting

- **Out of memory:** keep **Offload AR Stage** enabled, use tiled decode, or use FP8 on supported NVIDIA hardware.
- **Missing files offline:** allow the first download to complete before enabling **Offline / Local Files Only**.
- **Hash failure:** remove only the reported corrupt snapshot or file and download it again.
- **Wrong sampler or latent:** use **YuE2 Empty Latent**, **YuE2 Explicit Midpoint Sampler**, **YuE2 Linear Schedule**, and native `SamplerCustomAdvanced` together.
- **Need waveform editing or a reference recording:** convert the musical content to reviewed ABC with a separate tool; YuE2 does not accept source audio as generation conditioning.

## Original sources

- Official source: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- Model: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- VAE: [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)
- Current-ComfyUI compatibility fork: [endman100/YuE](https://github.com/endman100/YuE)

## License

The original ComfyUI integration code in this repository is released under the MIT License. YuE2 model weights are separate downloads licensed under CC BY-NC 4.0 and are not included in this repository. The YuE2 inference dependency and its bundled components retain their upstream terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or commercial use.
