from __future__ import annotations

import os
from pathlib import Path
import sys


NODE_DIR = Path(__file__).parents[1]
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", NODE_DIR.parents[1])).resolve()
if not (COMFYUI_ROOT / "comfy").is_dir():
    raise RuntimeError(
        "ComfyUI source not found. Install this repository below ComfyUI/custom_nodes "
        "or set COMFYUI_ROOT."
    )
sys.path.insert(0, str(COMFYUI_ROOT))
