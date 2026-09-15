from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile
import torch


NODE_DIR = Path(__file__).parents[1]
COMFY_ROOT = NODE_DIR.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "comfyui_yue2_parity",
    NODE_DIR / "__init__.py",
    submodule_search_locations=[str(NODE_DIR)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = sys.modules[f"{spec.name}.nodes"]

from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced  # noqa: E402
from yue2.protocol import Sampling  # noqa: E402


ABC_SCORE = """X:1
T:Parity Test
M:4/4
L:1/8
Q:1/4=100
K:C
CDEF GABc|cBAG FEDC|
"""

CASES = [
    {
        "name": "01_pop_vocal_seed_123",
        "style": "warm acoustic pop, clear female vocal, piano and strings",
        "lyrics": "[Verse]\nShine through the night",
        "cot": "off",
        "seed": 123,
        "abc": "",
        "semantic": Sampling(min_tokens=32, max_tokens=200),
        "full_decode": False,
    },
    {
        "name": "02_electronic_instrumental_seed_2026",
        "style": "instrumental progressive electronic, analog synth, deep bass, crisp drums",
        "lyrics": "[Instrumental]",
        "cot": "off",
        "seed": 2026,
        "abc": "",
        "semantic": Sampling(
            temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1,
            penalty_window=64, min_tokens=32, max_tokens=160,
        ),
        "full_decode": False,
    },
    {
        "name": "03_rock_vocal_seed_987654",
        "style": "energetic alternative rock, gritty male vocal, electric guitars and live drums",
        "lyrics": "[Verse]\nThe road keeps calling my name\n[Chorus]\nWe rise again",
        "cot": "off",
        "seed": 987654,
        "abc": "",
        "semantic": Sampling(
            temperature=1.2, top_p=0.98, top_k=200, repetition_penalty=1.15,
            penalty_window=50, min_tokens=32, max_tokens=128,
        ),
        "full_decode": False,
    },
    {
        "name": "04_auto_abc_full_seed_42",
        "style": "cinematic folk ballad, intimate vocal, acoustic guitar and cello",
        "lyrics": "[Verse]\nCarry the morning home",
        "cot": "full",
        "seed": 42,
        "abc": "",
        "abc_sampling": Sampling(
            temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005,
            penalty_window=100, min_tokens=32, max_tokens=128,
        ),
        "semantic": Sampling(min_tokens=32, max_tokens=128),
        "full_decode": False,
    },
    {
        "name": "05_supplied_abc_melody_seed_314159",
        "style": "bright chamber pop rearrangement, duet vocals, strings and piano",
        "lyrics": "[Verse]\nFollow this familiar line",
        "cot": "melody",
        "seed": 314159,
        "abc": ABC_SCORE,
        "semantic": Sampling(
            temperature=0.9, top_p=0.92, top_k=80, repetition_penalty=1.2,
            penalty_window=50, min_tokens=32, max_tokens=96,
        ),
        "full_decode": True,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampling_values(sampling):
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "repetition_penalty": sampling.repetition_penalty,
        "penalty_window": sampling.penalty_window,
        "min_tokens": sampling.min_tokens,
        "max_tokens": sampling.max_tokens,
    }


def node_plan(runtime, case):
    sampling = case.get("abc_sampling", Sampling())
    return nodes.YuE2PlanScore.execute(
        runtime, case["style"], case["lyrics"], case["cot"], case["seed"],
        -1.0, case["abc"], sampling.temperature, sampling.top_p, sampling.top_k,
        sampling.repetition_penalty, sampling.penalty_window,
        sampling.min_tokens, sampling.max_tokens,
    ).result[0]


def node_semantic(runtime, plan, sampling):
    return nodes.YuE2GenerateSemantic.execute(
        runtime, plan, sampling.temperature, sampling.top_p, sampling.top_k,
        sampling.repetition_penalty, sampling.penalty_window,
        sampling.min_tokens, sampling.max_tokens,
    ).result


def flattened(samples):
    if getattr(samples, "is_nested", False):
        samples = torch.cat(samples.unbind(), dim=-1)
    return samples.detach().float().cpu()


def metrics(reference, actual):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    difference = reference.astype(np.float64) - actual.astype(np.float64)
    noise_power = float(np.mean(difference**2))
    signal_power = float(np.mean(reference.astype(np.float64) ** 2))
    return {
        "shape_equal": True,
        "exact_equal": bool(np.array_equal(reference, actual)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": math.sqrt(noise_power),
        "snr_db": None if noise_power == 0 else 10 * math.log10(signal_power / noise_power),
    }


def run_case(model, runtime, case, output_dir):
    started = time.perf_counter()
    plan = node_plan(runtime, case)
    (conditioning,) = node_semantic(runtime, plan, case["semantic"])
    semantic = conditioning[0][1]["yue2_semantic"]
    pipe = runtime.pipeline()

    official_latents = pipe.synthesize(semantic)
    official_latents = torch.as_tensor(official_latents, dtype=torch.float32).T.unsqueeze(0)

    empty_latent = nodes.YuE2EmptyLatent.execute(conditioning)[0]
    sampled = SamplerCustomAdvanced.execute(
        RandomNoise.execute(case["seed"])[0],
        BasicGuider.execute(model, conditioning)[0],
        nodes.YuE2ExplicitMidpoint.execute()[0],
        nodes.YuE2LinearSchedule.execute(32)[0],
        empty_latent,
    )[0]
    comfyui_latents = flattened(sampled["samples"])

    official_audio = np.asarray(
        pipe.decode(official_latents, full=case["full_decode"]), dtype=np.float32
    )
    comfyui_audio_value = nodes.YuE2Decode.execute(
        runtime, sampled, case["full_decode"]
    )[0]
    comfyui_audio = comfyui_audio_value["waveform"][0].T.float().cpu().numpy()

    latent_metrics = metrics(official_latents.numpy(), comfyui_latents.numpy())
    audio_metrics = metrics(official_audio, comfyui_audio)
    passed = bool(
        latent_metrics.get("exact_equal") and audio_metrics.get("exact_equal")
    )

    case_dir = output_dir / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    official_path = case_dir / "official.wav"
    comfyui_path = case_dir / "comfyui.wav"
    difference_path = case_dir / "difference.wav"
    soundfile.write(official_path, official_audio, 48000, subtype="PCM_24")
    soundfile.write(comfyui_path, comfyui_audio, 48000, subtype="PCM_24")
    soundfile.write(
        difference_path, official_audio - comfyui_audio, 48000, subtype="FLOAT"
    )
    result = {
        "name": case["name"],
        "passed": passed,
        "settings": {
            "style": case["style"],
            "lyrics": case["lyrics"],
            "cot": case["cot"],
            "seed": case["seed"],
            "supplied_abc": bool(case["abc"]),
            "semantic_sampling": sampling_values(case["semantic"]),
            "ode_steps": 32,
            "full_decode": case["full_decode"],
        },
        "comparison_boundary": (
            "One YuE2 Generate Semantic output is shared by the official synthesize/decode "
            "branch and the ComfyUI SamplerCustomAdvanced/YuE2 Decode branch."
        ),
        "plan_type": type(plan).__name__,
        "semantic_type": type(semantic).__name__,
        "semantic_tokens": len(semantic.tokens),
        "latents": latent_metrics,
        "audio": audio_metrics,
        "sample_rate": 48000,
        "duration_seconds": len(official_audio) / 48000,
        "artifacts": {
            "official": str(official_path),
            "comfyui": str(comfyui_path),
            "difference": str(difference_path),
            "official_sha256": sha256(official_path),
            "comfyui_sha256": sha256(comfyui_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (case_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=COMFY_ROOT / "output" / "yue2_parity_5cases",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, runtime = nodes.YuE2ModelLoader.execute(
        "hugging_face", "m-a-p/YuE2-3B", "m-a-p/YuE2-Vae", "bfloat16",
        "cuda", 24.0, False, True, True, "", "", "",
    ).result
    started = time.perf_counter()
    results = []
    try:
        for case in CASES:
            results.append(run_case(model, runtime, case, args.output_dir))
    finally:
        if runtime._pipeline is not None:
            runtime._pipeline.close()

    summary = {
        "comparison": "official YuE2 synthesize/decode vs ComfyUI custom sampling graph",
        "precision": "bfloat16",
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "all_passed": all(result["passed"] for result in results),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
