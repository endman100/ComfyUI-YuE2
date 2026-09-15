# ComfyUI-YuE2

Generate complete songs from a style prompt and lyrics with [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), directly in ComfyUI.

[Install](#install) · [Quick start](#quick-start) · [Example workflows](example_workflows) · [Original YuE2](https://github.com/multimodal-art-projection/YuE) · [Model setup](docs/MODEL_SETUP.md)

## What it does

- Generates 48 kHz stereo songs from a style prompt and lyrics.
- Can create, supply, or edit an ABC score (text notation for melody and chords) before generation.
- Uses ComfyUI's standard sampling, audio preview, and audio save nodes where they fit.
- Downloads the official weights on first use, or loads them from local and `extra_model_paths.yaml` folders.
- Supports `auto`, BF16, FP16, FP32, and NVIDIA FP8 for the autoregressive model.

YuE2 regenerates music from text and a symbolic score. It does not edit a recording, continue source audio, separate stems, or clone a voice.

## Listen to generated samples

These are complete generations, not short excerpts. Compare the official YuE2 pipeline on the left with this ComfyUI package on the right. The waveform fills GitHub's fixed-size media player; it does not change the audio.

<details>
<summary><strong>▶ 01 · Official City Lights prompt · 0:58</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/01_official.mp4" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/01_comfyui.mp4" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 02 · Electronic instrumental · 3:03</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/02_official.mp4" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/02_comfyui.mp4" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 03 · Rock vocal · 1:10</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/03_official.mp4" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/03_comfyui.mp4" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 04 · Automatically planned score · 1:43</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/04_official.mp4" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/04_comfyui.mp4" controls></video></td>
  </tr>
</table>

</details>

<details>
<summary><strong>▶ 05 · Supplied melody rearrangement · 1:07</strong></summary>

<table>
  <tr><th>Official pipeline</th><th>ComfyUI node</th></tr>
  <tr>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/05_official.mp4" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/endman100/ComfyUI-YuE2/main/docs/assets/audio_comparisons/05_comfyui.mp4" controls></video></td>
  </tr>
</table>

</details>

## Install

Requires ComfyUI `>=0.35.0`, Python `>=3.10`, and about 7.8 GB for the model files. NVIDIA CUDA with BF16 is the recommended setup.

Search for [`yue2-native`](https://registry.comfy.org/nodes/yue2-native) in ComfyUI Manager. If it is not available there, install manually:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/endman100/ComfyUI-YuE2.git
cd ComfyUI-YuE2
python -m pip install -r requirements.txt
```

Restart ComfyUI. The first **YuE2 Model Loader** run downloads the required YuE2-3B and VAE files. For offline use, custom paths, and memory options, see [Model setup](docs/MODEL_SETUP.md).

## Quick start

1. Open **Workflow Templates → Custom Nodes → ComfyUI-YuE2**.
2. Load **YuE2 Text to Song (BF16)**.
3. Enter a **Style Prompt** and **Lyrics** in **YuE2 Plan Score**.
4. Queue the workflow, then listen with **Preview Audio** or save with **Save Audio**.

Start with the workflow defaults. Larger **Maximum Semantic Tokens** allows a longer song but also takes more time and VRAM.

## Example workflows

| Workflow | Choose it when you want to… |
| --- | --- |
| [`yue2_text_to_song_bf16.json`](example_workflows/yue2_text_to_song_bf16.json) | Generate a song from style and lyrics. Start here. |
| [`yue2_automatic_score.json`](example_workflows/yue2_automatic_score.json) | Let YuE2 create the melody and chords before generating audio. |
| [`yue2_abc_cover_rearrangement.json`](example_workflows/yue2_abc_cover_rearrangement.json) | Supply or edit an ABC score for a new arrangement. |
| [`yue2_text_to_song_fp8_nvidia.json`](example_workflows/yue2_text_to_song_fp8_nvidia.json) | Reduce AR-model memory use on a supported NVIDIA GPU. |

## Nodes

Most users only need to edit **YuE2 Model Loader**, **YuE2 Plan Score**, and optionally **YuE2 Plan Editor**. The remaining YuE2 nodes are already connected correctly in the templates.

| Node | What it is for |
| --- | --- |
| **YuE2 Model Loader** | Select model paths, data type, download, and offload settings. |
| **YuE2 Plan Score** | Turn style, lyrics, and an optional ABC score into a song plan. |
| **YuE2 Plan Editor** | Change the score or advanced plan fields without recreating the plan. |
| **YuE2 Generate Semantic** | Run YuE2's trained music-token model. |
| **YuE2 Empty Latent** | Create the empty acoustic canvas required by YuE2. |
| **YuE2 Explicit Midpoint Sampler** | Supply YuE2's sampler to ComfyUI's **SamplerCustomAdvanced**. |
| **YuE2 Linear Schedule** | Supply YuE2's sampling schedule to **SamplerCustomAdvanced**. |
| **YuE2 Decode** | Convert the sampled result to 48 kHz stereo audio. |

The workflows reuse ComfyUI's **RandomNoise**, **BasicGuider**, **SamplerCustomAdvanced**, **Preview Audio**, and **Save Audio** nodes.

## ABC score and Plan Editor

ABC is a readable text format for melody, rhythm, tempo, key, and chords.

- Choose `off` for the simplest text-to-song path.
- Choose `melody` to create or follow a melody.
- Choose `full` to create or follow melody and chord notation.

The ABC output is a readable preview; the same score is already stored inside `YUE2_PLAN`. Use **ABC Score Override** in **YuE2 Plan Editor** for normal score edits. **Advanced Overrides** is optional and accepts JSON fields such as `{"style":"jazz ballad","seed":42}`.

## Verified against official YuE2

Five quick regression cases and five complete listening cases produced exactly the same latents and raw audio as the official pipeline. The listening cases ran from 58 seconds to 3 minutes 3 seconds.

See the [reproducible validation report](docs/VALIDATION.md) for revisions, test method, metrics, and hashes.

## Troubleshooting

- **Out of memory:** keep **Offload AR Stage** enabled, use tiled decode, or use FP8 on supported NVIDIA hardware.
- **Missing files offline:** allow the first download to complete before enabling **Offline / Local Files Only**.
- **Hash failure:** remove only the reported corrupt snapshot or file and download it again.
- **Custom graph does not run:** keep **YuE2 Empty Latent**, **YuE2 Explicit Midpoint Sampler**, **YuE2 Linear Schedule**, and **SamplerCustomAdvanced** together.

## Project links

- Official source: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- Model: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- VAE: [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)
- ComfyUI compatibility fork: [endman100/YuE](https://github.com/endman100/YuE)

## License

The integration code is MIT licensed. Model weights and upstream runtime components keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
