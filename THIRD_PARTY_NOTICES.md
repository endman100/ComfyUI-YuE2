# Third-party notices

This repository contains an MIT-licensed ComfyUI integration. It does not
redistribute YuE2 model weights or the official `yue2_infer` wheel.

When explicitly run by the user, `install.py` downloads the current official
YuE2 inference wheel from the `m-a-p/YuE2-3B` Hugging Face repository, applies
the documented ComfyUI compatibility changes locally, and installs that local
wheel. The upstream wheel retains its own copyright notices, license files,
and third-party notices.

The YuE2 checkpoint weights are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The restriction
applies to the weights and is separate from this repository's MIT license.

- YuE2 model and inference package: https://huggingface.co/m-a-p/YuE2-3B
- YuE2 weight license: https://huggingface.co/m-a-p/YuE2-3B/blob/main/LICENSE
- YuE2 upstream third-party notices: https://huggingface.co/m-a-p/YuE2-3B/blob/main/THIRD_PARTY_NOTICES.md
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
