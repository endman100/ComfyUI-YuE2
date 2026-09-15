from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import importlib
import re
import threading
import time
from pathlib import Path

import torch

import folder_paths
import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models_base
from comfy import model_management


MODEL_ROOT = Path(folder_paths.models_dir) / "yue2"
folder_paths.add_model_folder_path("yue2", str(MODEL_ROOT), is_default=True)

HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
PRECISIONS = ("auto", "bfloat16", "float16", "float32", "fp8")


@dataclass(frozen=True)
class YuE2ModelConfig:
    source: str
    model: str
    vae: str
    precision: str
    device: str
    memory_budget_gib: float
    offload_ar: bool
    local_files_only: bool
    verify_hashes: bool
    revision: str | None
    vae_revision: str | None
    cache_dir: str | None


class YuE2ModelHandle:
    def __init__(self, config: YuE2ModelConfig):
        self.config = config
        self._pipeline = None
        self._lock = threading.Lock()

    def pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = open_pipeline(self.config)
        return self._pipeline

    def __getattr__(self, name):
        return getattr(self.config, name)


class _ComfyManagedModule(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device


class _YuE2LatentFormat(comfy.latent_formats.LatentFormat):
    latent_channels = 64
    latent_dimensions = 1


class _YuE2ModelConfig(comfy.supported_models_base.BASE):
    latent_format = _YuE2LatentFormat
    sampling_settings = {"shift": 1.0, "multiplier": 1000}
    memory_usage_factor = 0.0


class _YuE2FlowCarrier(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        raise RuntimeError("YuE2 MODEL requires the YuE2 Explicit Midpoint sampler")


class YuE2FlowModel(comfy.model_base.BaseModel):
    def __init__(self, handle, dtype, device):
        config = _YuE2ModelConfig({"disable_unet_model_creation": True})
        super().__init__(config, comfy.model_base.ModelType.FLOW, device=device)
        self.diffusion_model = _YuE2FlowCarrier(dtype)
        self.handle = handle

    def extra_conds(self, **kwargs):
        semantic = kwargs.get("yue2_semantic")
        if semantic is None:
            raise ValueError("YuE2 conditioning is required")
        if kwargs.get("yue2_runtime") is not self.handle:
            raise ValueError(
                "Connect MODEL and conditioning from the same YuE2 Model Loader"
            )
        return {"yue2_semantic": comfy.conds.CONDConstant(semantic)}


def create_model_patcher(handle):
    device = _resolve_device(handle.device)
    dtype, _quantization = _resolve_precision(handle.precision, device)
    offload_device = model_management.unet_offload_device()
    model = YuE2FlowModel(handle, dtype, offload_device)
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=device, offload_device=offload_device
    )
    patcher.model_options["yue2_handle"] = handle
    return patcher


def make_config(
    source: str,
    model: str,
    vae: str,
    precision: str,
    device: str,
    memory_budget_gib: float,
    offload_ar: bool,
    local_files_only: bool,
    verify_hashes: bool,
    revision: str,
    vae_revision: str,
    cache_dir: str,
) -> YuE2ModelHandle:
    if source not in {"hugging_face", "local"}:
        raise ValueError("source must be hugging_face or local")
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    if device not in {"auto", "cuda", "cpu", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    if not model.strip() or not vae.strip():
        raise ValueError("Both model and VAE locations are required")
    return YuE2ModelHandle(
        YuE2ModelConfig(
            source=source,
            model=model.strip(),
            vae=vae.strip(),
            precision=precision,
            device=device,
            memory_budget_gib=memory_budget_gib,
            offload_ar=offload_ar,
            local_files_only=local_files_only,
            verify_hashes=verify_hashes,
            revision=revision.strip() or None,
            vae_revision=vae_revision.strip() or None,
            cache_dir=cache_dir.strip() or None,
        )
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(model_management.get_torch_device())
    device = torch.device(value)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _resolve_precision(value: str, device: torch.device) -> tuple[torch.dtype, str]:
    if value == "fp8":
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError(
                "YuE2 FP8 requires an NVIDIA CUDA device with compute capability 8.9 or newer"
            )
        return torch.bfloat16, "fp8"
    if value == "auto":
        if device.type == "cuda":
            value = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif device.type == "mps":
            value = "float16"
        else:
            value = "float32"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value], "none"


def _resolve_local_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        candidate = path.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"YuE2 model directory not found: {candidate}")
        return str(candidate)

    for root_value in folder_paths.get_folder_paths("yue2"):
        root = Path(root_value).resolve()
        candidate = (root / path).resolve()
        if candidate.is_relative_to(root) and candidate.is_dir():
            return str(candidate)
    raise FileNotFoundError(
        f"YuE2 model directory not found under configured model paths: {value!r}"
    )


def _resolve_sources(config: YuE2ModelConfig) -> tuple[str, str]:
    if config.source == "local":
        return _resolve_local_path(config.model), _resolve_local_path(config.vae)
    for value in (config.model, config.vae):
        if not HF_REPO_ID.fullmatch(value):
            raise ValueError(f"Invalid Hugging Face repository id: {value!r}")
    return config.model, config.vae


def model_source_fingerprint(handle: YuE2ModelHandle):
    config = handle.config
    value = [repr(config)]
    if config.source != "local":
        return tuple(value)
    for source in _resolve_sources(config):
        root = Path(source)
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix in {".json", ".safetensors", ".tiktoken"}
                or path.name == "MANIFEST.sha256"
            )
        )
        value.append(
            (
                str(root),
                tuple(
                    (
                        str(path.relative_to(root)),
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                    )
                    for path in files
                ),
            )
        )
    return tuple(value)


def _load_yue2():
    module = importlib.import_module("yue2")
    version = tuple(
        int(part)
        for part in re.findall(r"\d+", getattr(module, "__version__", "0"))[:3]
    )
    if version < (0, 1, 6):
        raise RuntimeError("ComfyUI-YuE2 requires yue2-infer-comfyui 0.1.6 or newer")
    return module


@lru_cache(maxsize=1)
def _pipeline_type():
    yue2 = _load_yue2()
    base = yue2.YuE2Pipeline

    class DTypeYuE2Pipeline(base):
        def __init__(
            self,
            model_dir,
            vae_dir,
            *,
            device="auto",
            memory_budget_gib=24,
            backend="torch",
            generation_config=None,
            verify_hashes=True,
            vae_core_frames=None,
            quantization="none",
            offload_ar=False,
            progress=True,
            torch_dtype=torch.bfloat16,
        ):
            from yue2.protocol import GenerationConfig
            from yue2.storage import identity, model_identity, sha256_file
            from yue2.tokenization_yue2 import YuE2TextTokenizer

            if not isinstance(progress, bool):
                raise TypeError("progress must be True or False")
            if backend not in {"torch", "torch-eager", "vllm"}:
                raise ValueError("backend must be torch, torch-eager, or vllm")
            if quantization not in {"none", "fp8"}:
                raise ValueError("quantization must be none or fp8")
            if not 0 < memory_budget_gib:
                raise ValueError("memory_budget_gib must be positive")

            self.progress = progress
            self.device = torch.device(device)
            self.offload_device = model_management.unet_offload_device()
            self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
            self.backend, self.quantization = backend, quantization
            self.memory_budget_gib = float(memory_budget_gib)
            self.vae_core_frames = (
                vae_core_frames
                if vae_core_frames is not None
                else (512 if memory_budget_gib <= 12 else 1024)
            )
            self.offload_ar = offload_ar
            self.generation_config = generation_config or GenerationConfig()
            self.torch_dtype = torch_dtype
            self.tokenizer = YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")
            with self._status("Verifying model files"):
                self.weights = {
                    "mot": model_identity(self.model_dir, verify_hashes),
                    "vae": model_identity(self.vae_dir, verify_hashes),
                }
            runtime_files = sorted(
                Path(importlib.import_module("yue2").__file__).parent.glob("*.py")
            )
            self.runtime_sha256 = identity(
                {path.name: sha256_file(path) for path in runtime_files}
            )
            self._model, self._vae = None, None
            self._model_patcher, self._vae_patcher = None, None
            self.load_timing = {}

            if self.quantization == "fp8":
                if (
                    self.device.type != "cuda"
                    or not torch.cuda.is_available()
                    or torch.version.hip is not None
                ):
                    raise RuntimeError(
                        "YuE2 FP8 requires CUDA compute capability 8.9 or newer"
                    )
                if torch.cuda.get_device_capability(self.device) < (8, 9):
                    raise RuntimeError(
                        "YuE2 FP8 requires CUDA compute capability 8.9 or newer"
                    )
            if (
                self.torch_dtype == torch.bfloat16
                and self.device.type == "cuda"
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "This CUDA device does not support YuE2 BF16; select auto or float16"
                )

        def _load_model(self, for_nar=False):
            loading = self._model is None
            with self._status("Loading model") if loading else nullcontext():
                if self._model is None:
                    from yue2.modeling_yue2 import YuE2ForCausalLM

                    start = time.perf_counter()
                    self._model = YuE2ForCausalLM.from_pretrained(
                        self.model_dir,
                        local_files_only=True,
                        dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                    ).eval()
                    self._model.to(self.offload_device)
                    managed_model = _ComfyManagedModule(
                        self._model, self.offload_device
                    )
                    self._model_patcher = comfy.model_patcher.CoreModelPatcher(
                        managed_model,
                        load_device=self.device,
                        offload_device=self.offload_device,
                    )
                    model_management.archive_model_dtypes(self._model)
                    self.load_timing["mot_load_seconds"] = time.perf_counter() - start
                model_management.load_models_gpu(
                    [self._model_patcher], force_full_load=True
                )
                if self.quantization == "fp8" and not for_nar:
                    from yue2.quantization import prepare_fp8_ar

                    prepare_fp8_ar(self._model, self.device)
            return self._model

        def _load_vae(self):
            from yue2.modeling_vae import YuE2VAE

            if self.offload_ar and self._model_patcher is not None:
                model_management.unload_model_and_clones(self._model_patcher)
            with self._status("Loading audio decoder"):
                if self._vae is None:
                    self._vae = YuE2VAE.from_pretrained(
                        self.vae_dir,
                        decoder_only=True,
                        device=self.offload_device,
                        local_files_only=True,
                    )
                    managed_vae = _ComfyManagedModule(self._vae, self.offload_device)
                    self._vae_patcher = comfy.model_patcher.CoreModelPatcher(
                        managed_vae,
                        load_device=self.device,
                        offload_device=self.offload_device,
                    )
                    model_management.archive_model_dtypes(self._vae)
                model_management.load_models_gpu(
                    [self._vae_patcher], force_full_load=True
                )
            return self._vae

        def decode_audio(self, latents, *, full=False):
            vae = self._load_vae()

            z = torch.as_tensor(latents, dtype=torch.float32)
            if z.ndim == 2 and z.shape[1] == 64:
                z = z.T.unsqueeze(0)
            if z.ndim != 3 or z.shape[0] < 1 or z.shape[1] != 64:
                raise ValueError("Expected latents [T,64] or [B,64,T]")
            tiles = (
                1
                if full
                else (z.shape[-1] + self.vae_core_frames - 1) // self.vae_core_frames
            )
            with self._status("Decoding audio", total=tiles, unit="chunks") as status:

                def report(completed, total):
                    status.update(completed, total=total)
                    model_management.throw_exception_if_processing_interrupted()

                model_management.throw_exception_if_processing_interrupted()
                if full:
                    audio = vae.decode(z.to(self.device)).cpu()
                    status.update(1)
                else:
                    audio = vae.decode_tiled(
                        z,
                        core_frames=self.vae_core_frames,
                        halo_frames=16,
                        output_device="cpu",
                        on_progress=report,
                    )
            if not torch.isfinite(audio).all():
                raise ValueError("VAE produced non-finite audio")
            return audio.float().clamp(-1, 1)

        def decode(self, latents, *, full=False, vae=None):
            if vae is not None:
                raise ValueError("Use the VAE selected by YuE2 Model Loader")
            audio = self.decode_audio(latents, full=full)
            if audio.shape[0] != 1:
                raise ValueError("Song generation expects one latent batch")
            return audio[0].T.contiguous().numpy()

        def close(self):
            for patcher in (self._model_patcher, self._vae_patcher):
                if patcher is not None:
                    model_management.unload_model_and_clones(patcher)
            self._model, self._vae = None, None
            self._model_patcher, self._vae_patcher = None, None

        def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
            config = super().effective_config(request, abc_sampling, semantic_sampling)
            config["model_dtype"] = str(self.torch_dtype).removeprefix("torch.")
            config["memory_management"] = "comfyui"
            return config

    return DTypeYuE2Pipeline


def open_pipeline(config: YuE2ModelConfig):
    device = _resolve_device(config.device)
    dtype, quantization = _resolve_precision(config.precision, device)
    model, vae = _resolve_sources(config)
    cache_dir = config.cache_dir
    if config.source == "hugging_face" and cache_dir is None:
        cache_dir = str(MODEL_ROOT / "hub")
    pipeline_type = _pipeline_type()
    return pipeline_type.from_pretrained(
        model,
        vae=vae,
        revision=config.revision,
        vae_revision=config.vae_revision,
        local_files_only=config.local_files_only,
        cache_dir=cache_dir,
        device=device,
        memory_budget_gib=config.memory_budget_gib,
        quantization=quantization,
        offload_ar=config.offload_ar,
        verify_hashes=config.verify_hashes,
        progress=False,
        torch_dtype=dtype,
    )
