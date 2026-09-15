from __future__ import annotations

import json
import math
from itertools import pairwise

import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import torch
from comfy import model_management
from comfy_api.v0_0_2 import ComfyExtension, io
from typing_extensions import override

from .runtime import create_model_patcher, make_config, model_source_fingerprint

YuE2Runtime = io.Custom("YUE2_RUNTIME")
YuE2Plan = io.Custom("YUE2_PLAN")


def _plan_data(plan):
    return {
        "request": plan.request.to_dict(),
        "abc": plan.abc,
        "abc_ids": list(plan.abc_ids),
        "prefix": list(plan.prefix),
        "timing": dict(plan.timing),
        "truncated": bool(plan.truncated),
    }


def _semantic_from_conditioning(conditioning):
    for entry in conditioning:
        if len(entry) > 1 and "yue2_semantic" in entry[1]:
            return entry[1]["yue2_semantic"]
    raise ValueError(
        "YuE2 Empty Latent requires CONDITIONING from YuE2 Generate Semantic"
    )


def chunk_ranges(frames, prefix_tokens):
    from yue2.protocol import chunk_ranges as official_chunk_ranges

    return official_chunk_ranges(frames, prefix_tokens)


def _sampling(
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    penalty_window,
    min_tokens,
    max_tokens,
):
    from yue2.protocol import Sampling

    return Sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        penalty_window=penalty_window,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


def _official_noise_layout(noise):
    if noise.ndim != 3 or noise.shape[0] != 1 or noise.shape[1] != 64:
        raise ValueError("YuE2 noise must have shape [1,64,frames]")
    return noise.reshape(1, noise.shape[-1], 64).transpose(1, 2).contiguous()


def _explicit_midpoint(chunks, sigmas, velocity, on_step=None):
    output = []
    for chunk_index, source in enumerate(chunks):
        state = source
        for step, (start, end) in enumerate(pairwise(sigmas)):
            dt = start - end
            first = velocity(state, start, chunk_index)
            mid = state - first * (dt / 2)
            state = state - velocity(mid, (start + end) / 2, chunk_index) * dt
            if on_step is not None:
                on_step(chunk_index, step, state)
        output.append(state)
    return output


def _unpack_sampler_latents(packed, shapes):
    if len(shapes) == 1:
        return [packed.reshape(shapes[0])]
    return comfy.utils.unpack_latents(packed, shapes)


def _semantic_from_guider(model):
    try:
        cond = model.inner_model.conds["positive"]
        return cond[0]["model_conds"]["yue2_semantic"].cond
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Connect YuE2 Generate Semantic to BasicGuider conditioning"
        ) from exc


def _raw_time(sigma):
    value = float(sigma)
    if value <= 0:
        return -20.0
    if value >= 1:
        return 20.0
    return max(-20.0, min(20.0, math.log(value / (1.0 - value))))


def _sample_yue2(model, noise, sigmas, extra_args, callback, disable, **_kwargs):
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("YuE2 does not support latent inpainting")
    handle = extra_args.get("model_options", {}).get("yue2_handle")
    if handle is None:
        raise ValueError("YuE2 Explicit Midpoint requires a YuE2 MODEL")
    semantic = _semantic_from_guider(model)
    shapes = model.inner_model.inner_model.latent_shapes
    if shapes is None:
        shapes = [noise.shape]
    chunks = [
        _official_noise_layout(value)
        for value in _unpack_sampler_latents(noise, shapes)
    ]

    pipe = handle.pipeline()
    if pipe.quantization != "none" and pipe._model is not None:
        from yue2.quantization import restore_ar

        restore_ar(pipe._model)
    nar_model = pipe._load_model(for_nar=True)
    from yue2.nar import CachedNAR, Chunk
    from yue2.protocol import CODEC_OFFSET, MUSIC_END

    ranges = chunk_ranges(len(semantic.tokens), len(semantic.plan.prefix))
    if len(ranges) != len(chunks):
        raise ValueError("YuE2 latent chunks do not match the semantic token plan")
    progress = comfy.utils.ProgressBar(len(chunks) * (len(sigmas) - 1))
    completed = 0
    results = []
    for chunk_index, ((start, end), state) in enumerate(zip(ranges, chunks)):
        expected = end - start
        if state.shape[-1] != expected:
            raise ValueError(
                "YuE2 latent chunk length does not match the semantic token plan"
            )
        tokens = (
            semantic.plan.prefix
            + [value + CODEC_OFFSET for value in semantic.tokens[start:end]]
            + [MUSIC_END]
        )
        state = state.to(
            device=pipe.device, dtype=next(nar_model.vae2llm.parameters()).dtype
        )
        engine = CachedNAR(nar_model, Chunk(tokens, state[0].T))
        try:

            def velocity(value, sigma, _index, engine=engine):
                model_management.throw_exception_if_processing_interrupted()
                return engine.velocity(value[0].T, _raw_time(sigma)).T.unsqueeze(0)

            def on_step(_index, step, value):
                nonlocal completed
                completed += 1
                progress.update_absolute(completed)

            results.append(
                _explicit_midpoint([state], sigmas, velocity, on_step)[0].float()
            )
        finally:
            engine.close()
    if len(results) == 1:
        return results[0]
    packed, _shapes = comfy.utils.pack_latents(results)
    return packed


class YuE2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2ModelLoader",
            display_name="YuE2 Model Loader",
            category="audio/generation/yue2",
            description="Creates a lazy YuE2 runtime and a ComfyUI flow MODEL. Weights download/load only when a downstream YuE2 stage runs.",
            inputs=[
                io.Combo.Input(
                    "source",
                    options=["hugging_face", "local"],
                    default="hugging_face",
                    display_name="Weight Source",
                    tooltip="Download from Hugging Face, or load existing local model folders.",
                ),
                io.String.Input(
                    "model",
                    default="m-a-p/YuE2-3B",
                    display_name="Model ID / Path",
                    tooltip="Hugging Face repository ID or local YuE2 model directory.",
                ),
                io.String.Input(
                    "vae",
                    default="m-a-p/YuE2-Vae",
                    display_name="VAE ID / Path",
                    tooltip="Hugging Face repository ID or local YuE2 VAE directory.",
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "bfloat16", "float16", "float32", "fp8"],
                    default="auto",
                    display_name="Compute DType",
                    tooltip="Model compute precision. FP8 uses BF16 compute with FP8-quantized linear weights.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default="auto",
                    display_name="Device",
                    tooltip="Execution device. Auto follows ComfyUI's selected device.",
                    advanced=True,
                ),
                io.Float.Input(
                    "memory_budget_gib",
                    default=24.0,
                    min=4.0,
                    max=256.0,
                    step=1.0,
                    display_name="VRAM Budget (GiB)",
                    tooltip="Approximate GPU memory budget used to choose YuE2 loading and offload behavior.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "offload_ar",
                    default=True,
                    display_name="Offload AR Stage",
                    tooltip="Release the score and semantic language model before acoustic sampling to reduce peak VRAM.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "local_files_only",
                    default=False,
                    display_name="Offline / Local Files Only",
                    tooltip="Disable downloads and require every requested model file to exist locally.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "verify_hashes",
                    default=True,
                    display_name="Verify Downloaded Hashes",
                    tooltip="Verify available model hashes before loading.",
                    advanced=True,
                ),
                io.String.Input(
                    "revision",
                    default="",
                    display_name="Model Revision",
                    tooltip="Optional Hugging Face branch, tag, or commit. Blank uses the repository default.",
                    advanced=True,
                ),
                io.String.Input(
                    "vae_revision",
                    default="",
                    display_name="VAE Revision",
                    tooltip="Optional VAE branch, tag, or commit. Blank uses the repository default.",
                    advanced=True,
                ),
                io.String.Input(
                    "cache_dir",
                    default="",
                    display_name="Cache Directory",
                    tooltip="Optional model cache directory. Blank uses ComfyUI's YuE2 model folder.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="MODEL",
                    tooltip="ComfyUI flow model for BasicGuider and SamplerCustomAdvanced.",
                ),
                YuE2Runtime.Output(
                    display_name="YuE2 Runtime",
                    tooltip="Shared lazy runtime used by score, semantic, and decode nodes.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source,
        model,
        vae,
        precision,
        device,
        memory_budget_gib,
        offload_ar,
        local_files_only,
        verify_hashes,
        revision,
        vae_revision,
        cache_dir,
    ):
        handle = make_config(
            source,
            model,
            vae,
            precision,
            device,
            memory_budget_gib,
            offload_ar,
            local_files_only,
            verify_hashes,
            revision,
            vae_revision,
            cache_dir,
        )
        return io.NodeOutput(create_model_patcher(handle), handle)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            return model_source_fingerprint(make_config(**kwargs))
        except (OSError, TypeError, ValueError):
            return tuple(sorted((key, str(value)) for key, value in kwargs.items()))


class YuE2PlanScore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2PlanScore",
            display_name="YuE2 Plan Score",
            category="audio/generation/yue2",
            description="Builds a YuE2 song plan from style and lyrics. It can generate an ABC score or apply a supplied score for cover and rearrangement workflows.",
            inputs=[
                YuE2Runtime.Input(
                    "runtime",
                    display_name="YuE2 Runtime",
                    tooltip="Connect the runtime output from YuE2 Model Loader.",
                ),
                io.String.Input(
                    "style",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="Describe genre, instruments, mood, vocals, and arrangement...",
                    display_name="Style Prompt",
                    tooltip="Describe genre, instruments, vocal character, mood, and arrangement.",
                ),
                io.String.Input(
                    "lyrics",
                    multiline=True,
                    dynamic_prompts=True,
                    placeholder="[Verse]\nWrite song lyrics here...",
                    display_name="Lyrics",
                    tooltip="Song lyrics with section tags such as [Verse], [Chorus], and [Bridge].",
                ),
                io.Combo.Input(
                    "cot",
                    options=["full", "melody", "off"],
                    default="full",
                    display_name="Score Planning",
                    tooltip="Full plans harmony and melody; Melody plans the melody only; Off skips ABC score planning.",
                ),
                io.Int.Input(
                    "seed",
                    default=831001,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    display_name="Plan Seed",
                    tooltip="Seed for ABC score planning. This is separate from the acoustic RandomNoise seed.",
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=-1.0,
                    min=-1.0,
                    max=20.0,
                    step=0.01,
                    display_name="Plan CFG Scale",
                    tooltip="Classifier-free guidance for score planning. -1 uses YuE2's default behavior.",
                    advanced=True,
                ),
                io.String.Input(
                    "abc",
                    default="",
                    multiline=True,
                    placeholder="Optional ABC notation. Leave blank to generate a score...",
                    display_name="ABC Score (optional)",
                    tooltip="Human-readable music notation. Leave blank to generate it when Score Planning is Full or Melody; paste a score to preserve musical structure for a cover or rearrangement.",
                    advanced=True,
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    display_name="Score Temperature",
                    tooltip="Randomness used only while generating an ABC score.",
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p",
                    default=0.9,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    display_name="Score Top P",
                    tooltip="Nucleus-sampling threshold used only for ABC score generation.",
                    advanced=True,
                ),
                io.Int.Input(
                    "top_k",
                    default=30,
                    min=1,
                    max=10000,
                    display_name="Score Top K",
                    tooltip="Top-k sampling limit used only for ABC score generation.",
                    advanced=True,
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.005,
                    min=0.01,
                    max=10.0,
                    step=0.001,
                    display_name="Score Repetition Penalty",
                    tooltip="Discourages repeated ABC tokens during score generation.",
                    advanced=True,
                ),
                io.Int.Input(
                    "penalty_window",
                    default=100,
                    min=1,
                    max=100,
                    display_name="Score Penalty Window",
                    tooltip="Recent-token window used by the score repetition penalty.",
                    advanced=True,
                ),
                io.Int.Input(
                    "min_tokens",
                    default=32,
                    min=0,
                    max=16384,
                    display_name="Minimum Score Tokens",
                    tooltip="Minimum number of tokens allowed for generated ABC notation.",
                    advanced=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=4096,
                    min=1,
                    max=16384,
                    display_name="Maximum Score Tokens",
                    tooltip="Maximum number of tokens allowed for generated ABC notation.",
                    advanced=True,
                ),
            ],
            outputs=[
                YuE2Plan.Output(
                    display_name="Song Plan",
                    tooltip="Structured YuE2 plan consumed by Generate Semantic or Plan Editor. It already contains the ABC score.",
                ),
                io.String.Output(
                    display_name="ABC Score (text preview/export)",
                    tooltip="Readable copy of the score stored inside Song Plan. Generation does not require this socket; connect it only for preview, saving, or external text processing.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        runtime,
        style,
        lyrics,
        cot,
        seed,
        cfg_scale,
        abc,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        penalty_window,
        min_tokens,
        max_tokens,
    ):
        pipe = runtime.pipeline()
        progress = comfy.utils.ProgressBar(
            1 if cot == "off" or abc.strip() else max_tokens
        )
        completed = 0

        def on_token(_phase, _token):
            nonlocal completed
            completed += 1
            progress.update_absolute(completed)

        plan = pipe.plan(
            style=style,
            lyrics=lyrics,
            cot=cot,
            seed=seed,
            cfg_scale=None if cfg_scale < 0 else cfg_scale,
            abc=abc if abc.strip() else None,
            abc_sampling=_sampling(
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                penalty_window,
                min_tokens,
                max_tokens,
            ),
            cancelled=model_management.processing_interrupted,
            on_token=on_token,
        )
        return io.NodeOutput(plan, plan.abc or "")


class YuE2PlanEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2PlanEditor",
            display_name="YuE2 Plan Editor",
            category="audio/generation/yue2/advanced",
            description="Edits an existing YuE2 song plan. Use ABC Score Override for notation-only changes, or Advanced Overrides for JSON fields; tokenizer-derived data is rebuilt automatically.",
            inputs=[
                YuE2Runtime.Input(
                    "runtime",
                    display_name="YuE2 Runtime",
                    tooltip="Connect the same runtime used to create the source plan.",
                ),
                YuE2Plan.Input(
                    "plan",
                    display_name="Source Song Plan",
                    tooltip="Plan to edit. Blank override fields retain values from this plan.",
                ),
                io.String.Input(
                    "abc",
                    default="",
                    multiline=True,
                    placeholder="Optional replacement ABC score; blank keeps the current score...",
                    display_name="ABC Score Override (optional)",
                    tooltip="Paste edited ABC notation here. Leave blank to retain the score already stored in Source Song Plan.",
                ),
                io.String.Input(
                    "edits_json",
                    default="",
                    multiline=True,
                    placeholder='Optional JSON, e.g. {"style":"jazz ballad","seed":42}',
                    advanced=True,
                    display_name="Advanced Overrides (JSON, optional)",
                    tooltip=(
                        'Optional JSON object, for example {"style":"jazz ballad","seed":42}. '
                        "Supported fields: style, lyrics, cot, seed, cfg_scale, abc, and id. "
                        "A complete Plan Data object may also be pasted; derived token fields are ignored and rebuilt."
                    ),
                ),
            ],
            outputs=[
                YuE2Plan.Output(
                    display_name="Edited Song Plan",
                    tooltip="Updated YuE2 plan for Generate Semantic.",
                ),
                io.String.Output(
                    display_name="Edited ABC Score (text preview/export)",
                    tooltip="Readable copy of the edited score. It is not required by Generate Semantic.",
                ),
                io.Dict.Output(
                    display_name="Plan Data (DICT)",
                    tooltip="Plain editable metadata for inspection or generic ComfyUI dictionary processing; generated token fields are included for reference.",
                ),
            ],
        )

    @classmethod
    def execute(cls, runtime, plan, abc, edits_json):
        from yue2.protocol import SongRequest

        try:
            edits = json.loads(edits_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"YuE2 Plan Editor received invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(edits, dict):
            raise TypeError("YuE2 Plan Editor expects a JSON object")

        editable = {"style", "lyrics", "cot", "seed", "cfg_scale", "abc", "id"}
        derived = {"abc_ids", "prefix", "timing", "truncated"}
        request_data = plan.request.to_dict()
        request_data["abc"] = plan.abc

        nested = edits.get("request")
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError("YuE2 Plan Editor request must be a JSON object")
            unknown = set(nested) - editable
            if unknown:
                raise ValueError(
                    f"YuE2 Plan Editor cannot edit: {', '.join(sorted(unknown))}"
                )
            request_data.update(nested)

        unknown = set(edits) - editable - derived - {"request"}
        if unknown:
            raise ValueError(
                f"YuE2 Plan Editor cannot edit: {', '.join(sorted(unknown))}"
            )
        request_data.update({key: edits[key] for key in editable if key in edits})
        if abc.strip():
            request_data["abc"] = abc
        if request_data.get("cot") == "off":
            request_data["abc"] = None
        elif (
            not isinstance(request_data.get("abc"), str)
            or not request_data["abc"].strip()
        ):
            raise ValueError(
                "YuE2 Plan Editor requires nonempty ABC for melody/full plans"
            )

        edited = runtime.pipeline().plan(request=SongRequest(**request_data))
        return io.NodeOutput(edited, edited.abc or "", _plan_data(edited))


class YuE2GenerateSemantic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2GenerateSemantic",
            display_name="YuE2 Generate Semantic",
            category="audio/generation/yue2",
            description="Uses YuE2's autoregressive language model to generate semantic music tokens, then exposes standard ComfyUI CONDITIONING for BasicGuider.",
            inputs=[
                YuE2Runtime.Input(
                    "runtime",
                    display_name="YuE2 Runtime",
                    tooltip="Connect the runtime output from YuE2 Model Loader.",
                ),
                YuE2Plan.Input(
                    "plan",
                    display_name="Song Plan",
                    tooltip="Connect YuE2 Plan Score directly, or the Edited Song Plan from Plan Editor.",
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    display_name="Semantic Temperature",
                    tooltip="Randomness for semantic music-token generation.",
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    display_name="Semantic Top P",
                    tooltip="Nucleus-sampling threshold for semantic tokens.",
                    advanced=True,
                ),
                io.Int.Input(
                    "top_k",
                    default=100,
                    min=1,
                    max=10000,
                    display_name="Semantic Top K",
                    tooltip="Top-k sampling limit for semantic tokens.",
                    advanced=True,
                ),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=0.01,
                    max=10.0,
                    step=0.01,
                    display_name="Semantic Repetition Penalty",
                    tooltip="Discourages repeated semantic-token patterns.",
                    advanced=True,
                ),
                io.Int.Input(
                    "penalty_window",
                    default=50,
                    min=1,
                    max=100,
                    display_name="Semantic Penalty Window",
                    tooltip="Recent-token window used by the semantic repetition penalty.",
                    advanced=True,
                ),
                io.Int.Input(
                    "min_tokens",
                    default=200,
                    min=0,
                    max=24000,
                    display_name="Minimum Semantic Tokens",
                    tooltip="Minimum semantic sequence length.",
                    advanced=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=9000,
                    min=1,
                    max=24000,
                    display_name="Maximum Semantic Tokens",
                    tooltip="Maximum semantic sequence length and generation work limit.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="CONDITIONING",
                    tooltip="Carries YuE2 semantic tokens. Connect it to both BasicGuider and YuE2 Empty Latent.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        runtime,
        plan,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        penalty_window,
        min_tokens,
        max_tokens,
    ):
        progress = comfy.utils.ProgressBar(max_tokens)
        completed = 0

        def on_token(_phase, _token):
            nonlocal completed
            completed += 1
            progress.update_absolute(completed)

        semantic = runtime.pipeline().generate_semantic(
            plan,
            sampling=_sampling(
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                penalty_window,
                min_tokens,
                max_tokens,
            ),
            cancelled=model_management.processing_interrupted,
            on_token=on_token,
        )
        return io.NodeOutput(
            [[None, {"yue2_semantic": semantic, "yue2_runtime": runtime}]]
        )


class YuE2EmptyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2EmptyLatent",
            display_name="YuE2 Empty Latent",
            category="audio/generation/yue2",
            description="Creates the correctly sized empty YuE2 acoustic LATENT. It reads semantic-token length from YuE2 CONDITIONING and follows the official chunk boundaries.",
            inputs=[
                io.Conditioning.Input(
                    "conditioning",
                    display_name="YuE2 CONDITIONING",
                    tooltip="Connect the same CONDITIONING output from YuE2 Generate Semantic that also feeds BasicGuider.",
                )
            ],
            outputs=[
                io.Latent.Output(
                    display_name="Empty LATENT",
                    tooltip="Zero latent with YuE2's native chunk layout for SamplerCustomAdvanced.",
                )
            ],
        )

    @classmethod
    def execute(cls, conditioning):
        semantic = _semantic_from_conditioning(conditioning)
        ranges = chunk_ranges(len(semantic.tokens), len(semantic.plan.prefix))
        device = model_management.intermediate_device()
        chunks = [
            torch.zeros((1, 64, end - start), dtype=torch.float32, device=device)
            for start, end in ranges
        ]
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor(chunks)})


class YuE2ExplicitMidpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2ExplicitMidpoint",
            display_name="YuE2 Explicit Midpoint Sampler",
            category="sampling/custom_sampling/samplers",
            description="Provides YuE2's official chunk-major explicit-midpoint flow solver as a standard ComfyUI SAMPLER.",
            inputs=[],
            outputs=[
                io.Sampler.Output(
                    display_name="SAMPLER",
                    tooltip="Connect to SamplerCustomAdvanced. This solver preserves YuE2's official integration order.",
                )
            ],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(comfy.samplers.KSAMPLER(_sample_yue2))


class YuE2LinearSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2LinearSchedule",
            display_name="YuE2 Linear Schedule",
            category="sampling/custom_sampling/schedulers",
            description="Provides YuE2's exact linear flow schedule from 1 to 0, including both endpoints, as standard ComfyUI SIGMAS.",
            inputs=[
                io.Int.Input(
                    "steps",
                    default=32,
                    min=1,
                    max=10000,
                    display_name="Sampling Steps",
                    tooltip="Number of explicit-midpoint integration steps. The official YuE2 default is 32.",
                )
            ],
            outputs=[
                io.Sigmas.Output(
                    display_name="SIGMAS",
                    tooltip="Linear YuE2 flow schedule for SamplerCustomAdvanced.",
                )
            ],
        )

    @classmethod
    def execute(cls, steps):
        return io.NodeOutput(torch.linspace(1, 0, steps + 1, dtype=torch.float64))


class YuE2Decode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2Decode",
            display_name="YuE2 Decode",
            category="audio/generation/yue2",
            description="Decodes a sampled YuE2 LATENT into 48 kHz stereo AUDIO while preserving the official YuE2 output-level behavior.",
            inputs=[
                YuE2Runtime.Input(
                    "runtime",
                    display_name="YuE2 Runtime",
                    tooltip="Connect the same runtime used by the generation chain.",
                ),
                io.Latent.Input(
                    "latent",
                    display_name="Sampled LATENT",
                    tooltip="Connect the output from SamplerCustomAdvanced.",
                ),
                io.Boolean.Input(
                    "full_decode",
                    default=False,
                    display_name="Full Decode",
                    tooltip="Decode the complete latent at once. Disabled uses YuE2's tiled path to reduce peak VRAM.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Audio.Output(
                    display_name="AUDIO (48 kHz stereo)",
                    tooltip="Unnormalized YuE2 waveform for ComfyUI audio preview or save nodes.",
                )
            ],
        )

    @classmethod
    def execute(cls, runtime, latent, full_decode):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            samples = torch.cat(samples.unbind(), dim=-1)
        waveform = runtime.pipeline().decode_audio(samples, full=full_decode)
        return io.NodeOutput({"waveform": waveform, "sample_rate": 48000})


class YuE2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNodeABC]]:
        return [
            YuE2ModelLoader,
            YuE2PlanScore,
            YuE2PlanEditor,
            YuE2GenerateSemantic,
            YuE2EmptyLatent,
            YuE2ExplicitMidpoint,
            YuE2LinearSchedule,
            YuE2Decode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return YuE2Extension()
