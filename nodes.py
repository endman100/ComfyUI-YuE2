from __future__ import annotations

import math

import torch

import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy import model_management
from comfy_api.v0_0_2 import ComfyExtension, io
from typing_extensions import override

from .runtime import create_model_patcher, make_config, model_source_fingerprint


YuE2Runtime = io.Custom("YUE2_RUNTIME")
YuE2Plan = io.Custom("YUE2_PLAN")
YuE2Semantic = io.Custom("YUE2_SEMANTIC")


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
        for step, (start, end) in enumerate(zip(sigmas[:-1], sigmas[1:])):
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

            def velocity(value, sigma, _index):
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
                    "source", options=["hugging_face", "local"], default="hugging_face"
                ),
                io.String.Input("model", default="m-a-p/YuE2-3B"),
                io.String.Input("vae", default="m-a-p/YuE2-Vae"),
                io.Combo.Input(
                    "precision",
                    options=["auto", "bfloat16", "float16", "float32", "fp8"],
                    default="auto",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu", "mps"],
                    default="auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "memory_budget_gib",
                    default=24.0,
                    min=4.0,
                    max=256.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Boolean.Input("offload_ar", default=True, advanced=True),
                io.Boolean.Input("local_files_only", default=False, advanced=True),
                io.Boolean.Input("verify_hashes", default=True, advanced=True),
                io.String.Input("revision", default="", advanced=True),
                io.String.Input("vae_revision", default="", advanced=True),
                io.String.Input("cache_dir", default="", advanced=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                YuE2Runtime.Output(display_name="YuE2 runtime"),
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
            description="Creates or applies editable ABC notation for original generation, covers, and rearrangement.",
            inputs=[
                YuE2Runtime.Input("runtime"),
                io.String.Input("style", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "cot", options=["full", "melody", "off"], default="full"
                ),
                io.Int.Input(
                    "seed",
                    default=831001,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=-1.0,
                    min=-1.0,
                    max=20.0,
                    step=0.01,
                    advanced=True,
                ),
                io.String.Input("abc", default="", multiline=True, advanced=True),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p", default=0.9, min=0.01, max=1.0, step=0.01, advanced=True
                ),
                io.Int.Input("top_k", default=30, min=1, max=10000, advanced=True),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.005,
                    min=0.01,
                    max=10.0,
                    step=0.001,
                    advanced=True,
                ),
                io.Int.Input(
                    "penalty_window", default=100, min=1, max=100, advanced=True
                ),
                io.Int.Input("min_tokens", default=32, min=0, max=16384, advanced=True),
                io.Int.Input(
                    "max_tokens", default=4096, min=1, max=16384, advanced=True
                ),
            ],
            outputs=[
                YuE2Plan.Output(display_name="plan"),
                io.String.Output(display_name="ABC score"),
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


class YuE2GenerateSemantic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2GenerateSemantic",
            display_name="YuE2 Generate Semantic",
            category="audio/generation/yue2",
            description="Generates semantic music tokens and standard ComfyUI conditioning for BasicGuider.",
            inputs=[
                YuE2Runtime.Input("runtime"),
                YuE2Plan.Input("plan"),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "top_p", default=0.95, min=0.01, max=1.0, step=0.01, advanced=True
                ),
                io.Int.Input("top_k", default=100, min=1, max=10000, advanced=True),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.2,
                    min=0.01,
                    max=10.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "penalty_window", default=50, min=1, max=100, advanced=True
                ),
                io.Int.Input(
                    "min_tokens", default=200, min=0, max=24000, advanced=True
                ),
                io.Int.Input(
                    "max_tokens", default=9000, min=1, max=24000, advanced=True
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                YuE2Semantic.Output(display_name="semantic"),
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
            [[None, {"yue2_semantic": semantic, "yue2_runtime": runtime}]], semantic
        )


class YuE2EmptyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2EmptyLatent",
            display_name="YuE2 Empty Latent",
            category="audio/generation/yue2",
            inputs=[YuE2Semantic.Input("semantic")],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, semantic):
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
            display_name="YuE2 Explicit Midpoint",
            category="sampling/custom_sampling/samplers",
            description="YuE2's official chunk-major explicit-midpoint flow solver.",
            inputs=[],
            outputs=[io.Sampler.Output(display_name="sampler")],
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
            description="Exact YuE2 flow schedule from 1 to 0, including both endpoints.",
            inputs=[io.Int.Input("steps", default=32, min=1, max=10000)],
            outputs=[io.Sigmas.Output(display_name="sigmas")],
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
            description="Decodes standard YuE2 LATENT to unnormalized 48 kHz stereo AUDIO.",
            inputs=[
                YuE2Runtime.Input("runtime"),
                io.Latent.Input("latent"),
                io.Boolean.Input("full_decode", default=False, advanced=True),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
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
            YuE2GenerateSemantic,
            YuE2EmptyLatent,
            YuE2ExplicitMidpoint,
            YuE2LinearSchedule,
            YuE2Decode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return YuE2Extension()
