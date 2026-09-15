# Third-party notices

This repository contains an MIT-licensed ComfyUI integration. It does not
redistribute YuE2 model weights or the official `yue2_infer` wheel.

The install-time dependency `yue2-infer-comfyui` is pinned to an auditable
revision of the public `endman100/YuE` compatibility fork. The fork is based on
the official YuE2 0.1.6 source and retains its copyright notices, license files,
and third-party notices. It keeps ComfyUI in control of the Torch environment
and routes supported attention operations through ComfyUI's selected backend.

The YuE2 checkpoint weights are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The restriction
applies to the weights and is separate from this repository's MIT license.

- YuE2 model and inference package: https://huggingface.co/m-a-p/YuE2-3B
- ComfyUI compatibility fork: https://github.com/endman100/YuE/tree/codex/comfyui-compat
- YuE2 weight license: https://huggingface.co/m-a-p/YuE2-3B/blob/main/LICENSE
- YuE2 upstream third-party notices: https://huggingface.co/m-a-p/YuE2-3B/blob/main/THIRD_PARTY_NOTICES.md
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
