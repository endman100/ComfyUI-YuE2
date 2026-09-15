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

Each case places the **Official pipeline** and **ComfyUI node** outputs side by side for direct listening comparison. The samples are 48 kHz stereo AAC audio with no video track; the MP4 container is only used because GitHub embeds it in README files. Chrome may start the controls muted.

<details open>
<summary><strong>▶ 01 · Pop vocal · cot=off · seed 123</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/9413bdbb-a192-4a3c-8b7f-bda0212ff53a" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/092ab23c-803d-44fe-8efd-ac1e6d2a7da3" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 02 · Electronic instrumental · seed 2026</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/463c025b-268a-49d8-8df5-0c18dbde6419" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/6d332e0c-6fee-43f8-abb2-109301789ed2" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 03 · Rock vocal · high-temperature semantic sampling · seed 987654</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/2877d0a2-b503-41d8-8fbf-c032349f7d3d" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/025a56fc-b991-4add-9f86-4758941f5f4e" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 04 · Automatically planned full ABC score · seed 42</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/df99027e-6389-4da3-be08-a36bac5df374" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/8c7fe8dd-a390-44f8-b79a-b33134365bec" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 05 · Supplied ABC melody rearrangement · seed 314159</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f7a6e844-901b-4c32-804d-05f9b36573b6" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/fa2e322f-af92-4839-8dd0-bcf1b74bd7c0" controls></video></td>
  </tr>
</table>

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
