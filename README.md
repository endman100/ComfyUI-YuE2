# ComfyUI-YuE2

Composable ComfyUI V3 nodes for [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B). The node set keeps only YuE2-specific operations and uses ComfyUI's built-in noise, guider, sampling executor, audio preview, and audio save nodes.

## Install

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/endman100/ComfyUI-YuE2.git
cd ComfyUI-YuE2
python -m pip install -r requirements.txt
python install.py
```

`install.py` discovers the newest official `yue2_infer` wheel in the YuE2-3B repository and builds a local compatibility wheel. It keeps the installed ComfyUI Torch stack, relaxes only the upstream environment pins, and routes AR/NAR attention through ComfyUI's selected optimized backend. Restart ComfyUI after installation.

## Test

Run the test suite from this repository's root inside a configured ComfyUI Python environment:

```powershell
python -m pytest -q
```

## Workflow

Use these eight YuE2 nodes:

1. **YuE2 Model Loader** returns a standard `MODEL` and a shared `YUE2_RUNTIME`.
2. **YuE2 Plan Score** generates or accepts editable ABC notation.
3. **YuE2 Plan Editor** safely applies JSON edits to a plan, rebuilds tokenizer-derived fields, and exposes the result as a native `DICT`.
4. **YuE2 Generate Semantic** returns standard `CONDITIONING` containing the generated semantic-token payload.
5. **YuE2 Empty Latent** reads that same `CONDITIONING` and returns a standard `LATENT`, split at the official context boundaries.
6. **YuE2 Explicit Midpoint Sampler** returns a standard `SAMPLER` implementing the official chunk-major RK2 solver.
7. **YuE2 Linear Schedule** returns standard `SIGMAS`, exactly `1 → 0` with 32 steps by default.
8. **YuE2 Decode** converts the sampled standard `LATENT` to 48 kHz stereo `AUDIO` without peak, RMS, or loudness normalization; it only clamps out-of-range samples to `[-1, 1]`.

Every YuE2 node, input, and output includes an English description or tooltip in ComfyUI. Multiline text fields also show purpose-specific placeholders when empty. Hover a node title, socket, or widget for more detail.

Connect the native ComfyUI nodes as follows:

```text
YuE2 Model Loader.MODEL ───────────────┐
YuE2 Plan Score → [YuE2 Plan Editor] → YuE2 Generate Semantic       │
YuE2 Generate Semantic.CONDITIONING ───┬→ BasicGuider ───────────────┐
                                       └→ YuE2 Empty Latent ─────────┤
RandomNoise ──────────────────────────────────────────────────────────┤
YuE2 Explicit Midpoint Sampler ───────────────────────────────────────┤
YuE2 Linear Schedule ─────────────────────────────────────────────────┤
                                                                        └→ SamplerCustomAdvanced
SamplerCustomAdvanced.output → YuE2 Decode → Preview Audio / Save Audio
```

Use the same seed in **YuE2 Plan Score** and ComfyUI's **RandomNoise**. The custom sampler remaps ComfyUI's noise layout so the seeded CPU FP32 draw matches YuE2's official token-major noise.

**ABC score** is human-readable text notation for melody, rhythm, meter, key, tempo, and optional chord symbols. The score is stored inside `YUE2_PLAN`, so the separate **ABC Score (text preview/export)** string output does not need to be connected for generation. It is exposed only when you want to preview, copy, save, or process the notation with generic text nodes. In **YuE2 Plan Score**, leave **ABC Score (optional)** blank to let `full` or `melody` planning generate a score, or paste a score to preserve musical structure for a cover or rearrangement. `off` skips score planning.

**YuE2 Plan Editor** has two editing paths. For the common case, paste revised notation into **ABC Score Override (optional)**; leave it blank to retain the current score. For metadata changes, expand **Advanced Overrides (JSON, optional)** and enter an object such as `{"style":"jazz ballad","seed":42}`. Supported keys are `style`, `lyrics`, `cot`, `seed`, `cfg_scale`, `abc`, and `id`. The node also accepts its complete plan-data representation when copied back as JSON, but always discards and rebuilds `abc_ids`, `prefix`, timing, and truncation metadata. Use ComfyUI's **Convert Dictionary to String** when a JSON string is needed downstream.

## Example workflows

ComfyUI discovers the files in `example_workflows` automatically and lists them under **Workflow Templates → Custom Nodes → ComfyUI-YuE2**:

- `yue2_text_to_song_bf16`: short text-to-song generation using the native custom-sampling chain.
- `yue2_automatic_score`: generates a chord-annotated ABC plan before semantic and acoustic generation.
- `yue2_abc_cover_rearrangement`: uses editable supplied ABC notation and demonstrates **YuE2 Plan Editor** plus its native `DICT` output.
- `yue2_text_to_song_fp8_nvidia`: lower-memory FP8 autoregressive generation for supported NVIDIA GPUs; NAR synthesis remains BF16.

All Hugging Face examples keep model loading lazy and allow the loader to download only the requested YuE2 model artifacts on first execution. The FP8 example requires NVIDIA compute capability 8.9 or newer; use the BF16 workflow elsewhere.

## Models, downloading, and cache

- `hugging_face` accepts repository IDs. The default model and VAE are downloaded on the first downstream execution, not when the loader node is created. The default cache is `models/yue2/hub`.
- `local` accepts an absolute directory or a directory below a configured `models/yue2` root.
- `local_files_only` disables downloads and requires a complete local Hugging Face cache.
- Local model configuration, tokenizer, manifests, and weight timestamps participate in the loader fingerprint.
- The 3B model and VAE are wrapped by ComfyUI model patchers for cache reuse and accelerator offloading. With `offload_ar` enabled, decoding unloads the main model before loading the VAE.

## Precision

| Setting | Main model | VAE |
| --- | --- | --- |
| `auto` | BF16 on supported CUDA, FP16 on other CUDA/MPS, FP32 on CPU | FP32 |
| `bfloat16` | BF16 | FP32 |
| `float16` | FP16 | FP32 |
| `float32` | FP32 | FP32 |
| `fp8` | Official FP8 AR linears; exact BF16 weights restored for NAR | FP32 |

FP8 requires NVIDIA compute capability 8.9 or newer. It changes the autoregressive planning/token stages; the acoustic NAR flow stage uses restored BF16 weights.

## Supported YuE2 functions

- Original text-to-song generation: supported.
- `full`, `melody`, and `off` symbolic planning: supported.
- Supplied or edited ABC for cover and rearrangement: supported.
- Separate planning, semantic generation, flow sampling, and decode stages: supported.
- Full and bounded tiled VAE decode: supported.
- Audio reference conditioning, audio inpainting, continuation from audio, and stem editing: not provided by YuE2-3B's official inference interface.

For a cover, transcribe the source recording to ABC with an external tool such as SheetSage2, then connect or paste the score into **YuE2 Plan Score**. Editing means revising ABC, style, or lyrics and rendering again; it is not waveform inpainting.

The removed one-shot generator, sampling-config wrapper, synthesize wrapper, VAE encode node, and artifact saver duplicated functions now represented by the native graph or did not correspond to an official editing capability. Use ComfyUI's **Save Audio** for output.

## Validation

Run the unit tests with `python -m pytest -q`. On a CUDA system with the model files already cached, run `python tests/integration_parity.py` for five BF16 parity cases covering varied seeds and sampling settings, vocal and instrumental prompts, automatic ABC planning, supplied ABC rearrangement, tiled decode, and full decode.

The parity suite gives the official `YuE2Pipeline.synthesize/decode` path and the ComfyUI `SamplerCustomAdvanced/YuE2 Decode` path the same semantic tokens, then requires exact latent and float32 audio equality. Sharing the semantic input is intentional: independently repeating stochastic autoregressive generation tests CUDA/RNG repeatability rather than the custom flow sampler. Reports and A/B WAV files are written under `output/yue2_parity_5cases`.

## Compatibility

Tested against ComfyUI 0.35.0's `comfy_api.v0_0_2`, frontend 1.51.10, Python 3.12, PyTorch 2.14.0 + CUDA 13.0, and the official YuE2 inference 0.1.5 compatibility build on an NVIDIA RTX 5090. `v0_0_2` is the latest numbered V3 API in this release but is still marked experimental upstream. The node package requires ComfyUI 0.35.0 or newer.

## License

The original ComfyUI integration code in this repository is released under the MIT License. YuE2 model weights are separate downloads licensed under CC BY-NC 4.0 and are not included in this repository. The official inference wheel and its bundled components retain their upstream terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or commercial use.
