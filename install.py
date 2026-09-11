from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve
import zipfile


REPOSITORY = "m-a-p/YuE2-3B"
MODEL_API = f"https://huggingface.co/api/models/{REPOSITORY}"
WHEEL_PATTERN = re.compile(r"^yue2_infer-(\d+)\.(\d+)\.(\d+)-py3-none-any\.whl$")
FORK_REQUIREMENTS = {
    "torch": "torch>=2.10.0",
    "transformers": "transformers>=4.57.6,<5",
    "huggingface-hub": "huggingface-hub>=0.36.2,<1",
    "safetensors": "safetensors>=0.7.0",
    "tiktoken": "tiktoken>=0.12.0",
    "numpy": "numpy>=1.26.0",
    "soundfile": "soundfile>=0.13.1",
    "accelerate": "accelerate>=1.12.0",
}


def latest_upstream_wheel() -> tuple[str, str]:
    with urlopen(MODEL_API) as response:
        files = json.load(response)["siblings"]
    candidates = []
    for item in files:
        name = item["rfilename"]
        match = WHEEL_PATTERN.fullmatch(name)
        if match:
            candidates.append((tuple(map(int, match.groups())), name))
    if not candidates:
        raise RuntimeError(f"No yue2-infer wheel found in {REPOSITORY}")
    version, filename = max(candidates)
    return ".".join(map(str, version)), filename


def _requirement_name(line: str) -> str | None:
    if not line.startswith("Requires-Dist: "):
        return None
    value = line.removeprefix("Requires-Dist: ")
    return re.split(r"[ (<>=!~;\[]", value, maxsplit=1)[0].lower().replace("_", "-")


def _rewrite_metadata(metadata: str, upstream_version: str, fork_version: str) -> str:
    output = []
    replaced = set()
    for line in metadata.splitlines():
        if line == "Name: yue2-infer":
            output.append("Name: yue2-infer-comfyui")
        elif line == f"Version: {upstream_version}":
            output.append(f"Version: {fork_version}")
        elif (name := _requirement_name(line)) in FORK_REQUIREMENTS:
            if name not in replaced:
                output.append(f"Requires-Dist: {FORK_REQUIREMENTS[name]}")
                replaced.add(name)
        else:
            output.append(line)
    missing = set(FORK_REQUIREMENTS) - replaced
    if missing:
        raise RuntimeError(
            f"Upstream dependency metadata changed: missing {sorted(missing)}"
        )
    return "\n".join(output) + "\n"


def _record_row(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root).as_posix()
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        .rstrip(b"=")
        .decode()
    )
    return [relative, f"sha256={digest}", str(path.stat().st_size)]


def _patch_comfyui_attention(root: Path) -> None:
    modeling_path = root / "yue2" / "modeling_yue2.py"
    modeling = modeling_path.read_text(encoding="utf-8")
    import_anchor = "from transformers.modeling_outputs import CausalLMOutputWithPast\n"
    if import_anchor not in modeling:
        raise RuntimeError("YuE2 modeling imports changed")
    modeling = modeling.replace(
        import_anchor,
        import_anchor
        + "from comfy.ldm.modules.attention import optimized_attention_for_device\n",
        1,
    )
    start = modeling.index("def sdpa(")
    end = modeling.index("\n\ndef _causal_mask", start)
    comfy_sdpa = '''def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
    """Route YuE2 attention through ComfyUI's selected attention backend."""
    grouped = query.shape[1] != key.shape[1]
    if is_causal:
        query_length, key_length = query.shape[-2], key.shape[-2]
        offset = key_length - query_length
        visible = torch.arange(key_length, device=query.device)[None, :] <= (
            torch.arange(query_length, device=query.device)[:, None] + offset
        )
        if attn_mask is None:
            attn_mask = visible
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask & visible
        else:
            attn_mask = attn_mask.masked_fill(~visible, float("-inf"))
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = torch.zeros(attn_mask.shape, dtype=query.dtype, device=query.device).masked_fill(
            ~attn_mask, float("-inf")
        )
    attention = optimized_attention_for_device(
        query.device,
        mask=attn_mask is not None,
        small_input=query.shape[-2] <= 1,
    )
    return attention(
        query,
        key,
        value,
        query.shape[1],
        mask=attn_mask,
        skip_reshape=True,
        skip_output_reshape=True,
        enable_gqa=grouped,
    )
'''
    modeling_path.write_text(
        modeling[:start] + comfy_sdpa + modeling[end:], encoding="utf-8"
    )

    nar_path = root / "yue2" / "nar.py"
    nar = nar_path.read_text(encoding="utf-8")
    protocol_import = "from .protocol import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, chunk_ranges\n"
    if protocol_import not in nar:
        raise RuntimeError("YuE2 NAR imports changed")
    nar = nar.replace(
        protocol_import, protocol_import + "from .modeling_yue2 import sdpa\n", 1
    )
    native_call = """F.scaled_dot_product_attention(
                query[..., start:end, :], used_key, used_value,
                attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped,
            )"""
    comfy_call = """sdpa(
                query[..., start:end, :], used_key, used_value,
                attn_mask=mask, is_causal=causal and start == 0,
            )"""
    if native_call not in nar:
        raise RuntimeError("YuE2 NAR attention implementation changed")
    nar_path.write_text(nar.replace(native_call, comfy_call, 1), encoding="utf-8")

    cuda_graph_path = root / "yue2" / "cuda_graph.py"
    cuda_graph = cuda_graph_path.read_text(encoding="utf-8")
    flash_probe = "flash = fused and config.head_dim <= 256 and hasattr("
    if flash_probe not in cuda_graph:
        raise RuntimeError("YuE2 CUDA graph attention probe changed")
    cuda_graph_path.write_text(
        cuda_graph.replace(
            flash_probe,
            "flash = fused and torch.backends.cuda.is_flash_attention_available() and config.head_dim <= 256 and hasattr(",
            1,
        ),
        encoding="utf-8",
    )


def build_compat_wheel(
    upstream_wheel: Path, destination: Path, upstream_version: str
) -> Path:
    fork_version = f"{upstream_version}.post1"
    unpacked = destination / "unpacked"
    with zipfile.ZipFile(upstream_wheel) as archive:
        archive.extractall(unpacked)

    dist_info_dirs = list(unpacked.glob("yue2_infer-*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError("Unexpected yue2-infer wheel layout")
    old_dist_info = dist_info_dirs[0]
    metadata = (old_dist_info / "METADATA").read_text(encoding="utf-8")
    new_dist_info = unpacked / f"yue2_infer_comfyui-{fork_version}.dist-info"
    old_dist_info.rename(new_dist_info)
    (new_dist_info / "METADATA").write_text(
        _rewrite_metadata(metadata, upstream_version, fork_version),
        encoding="utf-8",
    )
    _patch_comfyui_attention(unpacked)

    record = new_dist_info / "RECORD"
    rows = [
        _record_row(path, unpacked)
        for path in sorted(unpacked.rglob("*"))
        if path.is_file() and path != record
    ]
    rows.append([record.relative_to(unpacked).as_posix(), "", ""])
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    record.write_text(buffer.getvalue(), encoding="utf-8", newline="")

    output = destination / f"yue2_infer_comfyui-{fork_version}-py3-none-any.whl"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(unpacked.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(unpacked).as_posix())
    return output


def install() -> None:
    upstream_version, filename = latest_upstream_wheel()
    with tempfile.TemporaryDirectory(prefix="yue2-comfyui-") as temp_value:
        temp = Path(temp_value)
        upstream = temp / filename
        url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{quote(filename)}"
        print(f"Downloading YuE2 inference {upstream_version} from {REPOSITORY}")  # noqa: T201
        urlretrieve(url, upstream)
        wheel = build_compat_wheel(upstream, temp, upstream_version)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "yue2-infer",
                "yue2-infer-comfyui",
            ]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)]
        )


if __name__ == "__main__":
    install()
