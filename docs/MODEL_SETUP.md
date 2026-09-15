# Model setup

This page covers weight discovery, download behavior, precision, and memory settings for ComfyUI-YuE2.

## Automatic download

Set **Weight Source** to `hugging_face`. The defaults are:

- Main model: `m-a-p/YuE2-3B`
- VAE: `m-a-p/YuE2-Vae`

Creating **YuE2 Model Loader** does not download or load weights. The first downstream YuE2 stage resolves both snapshots through `huggingface_hub.snapshot_download`. The allowlist covers model configuration, tokenizer, manifest, model code, documented notices, and Safetensors weights. **Verify Downloaded Hashes** checks the resolved manifest. Hugging Face partial-file handling prevents an interrupted transfer from appearing complete.

The default cache is:

```text
ComfyUI/models/yue2/hub/
```

Enable **Offline / Local Files Only** only after both snapshots are complete. Configure any Hugging Face credentials outside the workflow; do not store tokens in workflow JSON.

## Local models

Set **Weight Source** to `local`. **Model ID / Path** and **VAE ID / Path** may be absolute directories or relative directories inside a registered `yue2` model root.

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

For this layout, use `YuE2-3B` and `YuE2-Vae` as the relative path inputs. Paths are resolved and validated again when opened. Local configuration, tokenizer, manifests, weight sizes, and modification timestamps participate in the loader fingerprint.

## Extra model paths

The package registers the `yue2` model category with ComfyUI. Add a shared root to `extra_model_paths.yaml`, then restart ComfyUI:

```yaml
shared_models:
  base_path: D:/AI/models
  yue2: yue2
```

This discovers models below `D:/AI/models/yue2/`. Relative local paths must remain inside a configured `yue2` root. **Cache Directory** can separately override the Hugging Face cache used by `hugging_face` mode.

## Precision

| Loader setting | Main AR and score/semantic stages | NAR acoustic stage | VAE | Hardware note |
| --- | --- | --- | --- | --- |
| `auto` | BF16 on supported CUDA, FP16 on other CUDA or MPS, FP32 on CPU | Same compute dtype | FP32 | Recommended |
| `bfloat16` | BF16 | BF16 | FP32 | CUDA device must support BF16 |
| `float16` | FP16 | FP16 | FP32 | Use where BF16 is unavailable |
| `float32` | FP32 | FP32 | FP32 | Highest memory use; CPU-compatible but slow |
| `fp8` | BF16 compute with official FP8-quantized AR linear weights | Exact BF16 weights restored | FP32 | NVIDIA compute capability 8.9+ |

## Memory behavior

The main model and VAE use ComfyUI `CoreModelPatcher` objects. **Offload AR Stage** unloads the main language model before VAE loading. Disable **Full Decode** for bounded tiled decoding with overlap; enable full decode only when the complete official decode path fits in memory.

The tested YuE2-3B and default VAE snapshots require approximately 7.8 GB of storage, plus runtime memory and cache space.
