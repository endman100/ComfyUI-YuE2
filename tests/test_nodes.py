from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yue2.modeling_yue2
from yue2.pipeline import SymbolicPlan
from yue2.protocol import SongRequest

NODE_DIR = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "comfyui_yue2",
    NODE_DIR / "__init__.py",
    submodule_search_locations=[str(NODE_DIR)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules["comfyui_yue2.nodes"]
runtime = sys.modules["comfyui_yue2.runtime"]


def loader_values(**overrides):
    values = {
        "source": "hugging_face",
        "model": "m-a-p/YuE2-3B",
        "vae": "m-a-p/YuE2-Vae",
        "precision": "auto",
        "device": "auto",
        "memory_budget_gib": 24.0,
        "offload_ar": True,
        "local_files_only": False,
        "verify_hashes": True,
        "revision": "",
        "vae_revision": "",
        "cache_dir": "",
    }
    values.update(overrides)
    return values


def loader(**overrides):
    return nodes.YuE2ModelLoader.execute(**loader_values(**overrides)).result


def fake_semantic(frames=8, prefix_tokens=4, seed=7):
    request = SimpleNamespace(seed=seed)
    plan = SimpleNamespace(prefix=list(range(prefix_tokens)), request=request)
    return SimpleNamespace(
        plan=plan, tokens=list(range(frames)), truncated=False, timing={}
    )


def test_v3_extension_registers_only_minimal_composable_nodes():
    extension = asyncio.run(PACKAGE.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [
        nodes.YuE2ModelLoader,
        nodes.YuE2PlanScore,
        nodes.YuE2PlanEditor,
        nodes.YuE2GenerateSemantic,
        nodes.YuE2EmptyLatent,
        nodes.YuE2ExplicitMidpoint,
        nodes.YuE2LinearSchedule,
        nodes.YuE2Decode,
    ]
    for removed in (
        "YuE2SamplingConfig",
        "YuE2Generate",
        "YuE2Synthesize",
        "YuE2VAEEncode",
        "YuE2SaveArtifacts",
    ):
        assert not hasattr(nodes, removed)


def test_yue2_nodes_expose_english_ui_help():
    node_types = [
        nodes.YuE2ModelLoader,
        nodes.YuE2PlanScore,
        nodes.YuE2PlanEditor,
        nodes.YuE2GenerateSemantic,
        nodes.YuE2EmptyLatent,
        nodes.YuE2ExplicitMidpoint,
        nodes.YuE2LinearSchedule,
        nodes.YuE2Decode,
    ]
    for node_type in node_types:
        schema = node_type.define_schema()
        assert schema.description
        assert all(value.display_name and value.tooltip for value in schema.inputs)
        assert all(value.display_name and value.tooltip for value in schema.outputs)

    plan_schema = nodes.YuE2PlanScore.define_schema()
    plan_inputs = {value.id: value for value in plan_schema.inputs}
    assert plan_inputs["style"].display_name == "Style Prompt"
    assert plan_inputs["style"].placeholder.startswith("Describe genre")
    assert plan_inputs["lyrics"].display_name == "Lyrics"
    assert plan_inputs["lyrics"].placeholder.startswith("[Verse]")
    assert plan_inputs["abc"].display_name == "ABC Score (optional)"
    assert plan_inputs["abc"].placeholder.startswith("Optional ABC notation")
    assert plan_schema.outputs[1].display_name == "ABC Score (text preview/export)"

    editor_schema = nodes.YuE2PlanEditor.define_schema()
    editor_inputs = {value.id: value for value in editor_schema.inputs}
    assert editor_inputs["abc"].display_name == "ABC Score Override (optional)"
    assert editor_inputs["abc"].placeholder.startswith("Optional replacement ABC")
    assert editor_inputs["edits_json"].display_name == (
        "Advanced Overrides (JSON, optional)"
    )
    assert editor_inputs["edits_json"].default == ""
    assert editor_inputs["edits_json"].placeholder.startswith("Optional JSON")


def test_example_workflows_are_current_composable_ui_graphs():
    workflow_dir = NODE_DIR / "example_workflows"
    expected = {
        "yue2_text_to_song_bf16.json",
        "yue2_automatic_score.json",
        "yue2_abc_cover_rearrangement.json",
        "yue2_text_to_song_fp8_nvidia.json",
    }
    assert {path.name for path in workflow_dir.glob("*.json")} == expected

    required_types = {
        "YuE2ModelLoader",
        "YuE2PlanScore",
        "YuE2GenerateSemantic",
        "YuE2EmptyLatent",
        "RandomNoise",
        "BasicGuider",
        "YuE2ExplicitMidpoint",
        "YuE2LinearSchedule",
        "SamplerCustomAdvanced",
        "YuE2Decode",
        "SaveAudioAdvanced",
        "Note",
    }
    removed_types = {
        "YuE2SamplingConfig",
        "YuE2Generate",
        "YuE2Synthesize",
        "YuE2VAEEncode",
        "YuE2SaveArtifacts",
    }
    for path in workflow_dir.glob("*.json"):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert workflow["extra"]["frontendVersion"] == "1.51.10"
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        node_types = {node["type"] for node in workflow["nodes"]}
        assert required_types.issubset(node_types)
        if path.name == "yue2_abc_cover_rearrangement.json":
            assert node_types == required_types | {"PreviewAny", "YuE2PlanEditor"}
        elif path.name == "yue2_automatic_score.json":
            assert node_types == required_types | {"PreviewAny"}
        else:
            assert node_types == required_types
        assert node_types.isdisjoint(removed_types)

        semantic_node = next(
            node for node in workflow["nodes"] if node["type"] == "YuE2GenerateSemantic"
        )
        empty_latent = next(
            node for node in workflow["nodes"] if node["type"] == "YuE2EmptyLatent"
        )
        assert [output["type"] for output in semantic_node["outputs"]] == [
            "CONDITIONING"
        ]
        assert empty_latent["inputs"][0]["type"] == "CONDITIONING"

        note_titles = {
            node.get("title") for node in workflow["nodes"] if node["type"] == "Note"
        }
        assert {
            "YuE2 Model Loader — setup",
            "YuE2 Plan Score — inputs",
            "YuE2 Generate Semantic — conditioning",
            "YuE2 Linear Schedule — steps",
            "YuE2 Decode — audio",
        }.issubset(note_titles)
        assert all("Empty Latent" not in title for title in note_titles)
        assert all("Explicit Midpoint" not in title for title in note_titles)
        if path.name == "yue2_abc_cover_rearrangement.json":
            assert "YuE2 Plan Editor — overrides" in note_titles

        nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
        for link_id, source, source_slot, target, target_slot, data_type in workflow[
            "links"
        ]:
            assert nodes_by_id[source]["outputs"][source_slot]["type"] == data_type
            assert nodes_by_id[target]["inputs"][target_slot]["type"] in {
                data_type,
                "*",
            }
            assert link_id in nodes_by_id[source]["outputs"][source_slot]["links"]
            assert nodes_by_id[target]["inputs"][target_slot]["link"] == link_id

        loader = next(
            node for node in workflow["nodes"] if node["type"] == "YuE2ModelLoader"
        )
        assert loader["widgets_values_named"]["source"] == "hugging_face"
        assert loader["widgets_values_named"]["local_files_only"] is False
        plan = next(
            node for node in workflow["nodes"] if node["type"] == "YuE2PlanScore"
        )
        assert plan["title"] == "YuE2 Plan Score — Style / Lyrics / ABC Score"
        assert [output["name"] for output in plan["outputs"]] == [
            "Song Plan",
            "ABC Score (text preview/export)",
        ]
        sampler = next(
            node for node in workflow["nodes"] if node["type"] == "YuE2LinearSchedule"
        )
        assert sampler["widgets_values_named"]["steps"] == 32

    fp8 = json.loads((workflow_dir / "yue2_text_to_song_fp8_nvidia.json").read_text())
    fp8_loader = next(
        node for node in fp8["nodes"] if node["type"] == "YuE2ModelLoader"
    )
    assert fp8_loader["widgets_values_named"]["precision"] == "fp8"
    cover = json.loads((workflow_dir / "yue2_abc_cover_rearrangement.json").read_text())
    cover_plan = next(
        node for node in cover["nodes"] if node["type"] == "YuE2PlanScore"
    )
    assert cover_plan["widgets_values_named"]["abc"].startswith("X:1\n")
    cover_editor = next(
        node for node in cover["nodes"] if node["type"] == "YuE2PlanEditor"
    )
    assert cover_editor["title"] == "YuE2 Plan Editor — ABC Override / JSON Overrides"
    assert [input_["name"] for input_ in cover_editor["inputs"]] == [
        "runtime",
        "plan",
    ]
    assert [output["name"] for output in cover_editor["outputs"]] == [
        "Edited Song Plan",
        "Edited ABC Score (text preview/export)",
        "Plan Data (DICT)",
    ]
    assert cover_editor["widgets_values_named"] == {"abc": "", "edits_json": ""}


def test_loader_returns_standard_model_and_shared_lazy_runtime(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "open_pipeline",
        lambda _config: pytest.fail("loader opened the pipeline"),
    )
    monkeypatch.setattr(runtime, "_resolve_device", lambda _value: torch.device("cpu"))
    model, handle = loader(precision="float16")
    assert model.model.handle is handle
    assert model.model.model_type.name == "FLOW"
    assert model.get_model_object("model_sampling").sigma_max == 1
    assert handle.precision == "float16"
    assert handle.revision is None
    assert handle.cache_dir is None


def test_model_source_fingerprint_changes_when_local_weights_change(tmp_path):
    model_dir = tmp_path / "YuE2-3B"
    vae_dir = tmp_path / "YuE2-Vae"
    model_dir.mkdir()
    vae_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"first")
    _, handle = loader(source="local", model=str(model_dir), vae=str(vae_dir))
    before = runtime.model_source_fingerprint(handle)
    weights.write_bytes(b"second-version")
    assert runtime.model_source_fingerprint(handle) != before


def test_hugging_face_source_rejects_non_repo_paths():
    _, handle = loader(model="not-a-repo-id")
    with pytest.raises(ValueError, match="Invalid Hugging Face repository id"):
        runtime._resolve_sources(handle)


@pytest.mark.parametrize(
    ("precision", "expected_dtype", "quantization"),
    [
        ("bfloat16", torch.bfloat16, "none"),
        ("float16", torch.float16, "none"),
        ("float32", torch.float32, "none"),
        ("fp8", torch.bfloat16, "fp8"),
    ],
)
def test_precision_mapping(precision, expected_dtype, quantization):
    device = torch.device("cuda" if precision == "fp8" else "cpu")
    assert runtime._resolve_precision(precision, device) == (
        expected_dtype,
        quantization,
    )


def test_open_pipeline_forwards_precision_and_default_cache(monkeypatch):
    captured = {}

    class FakePipelineType:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            return "pipeline"

    monkeypatch.setattr(runtime, "_pipeline_type", lambda: FakePipelineType)
    monkeypatch.setattr(runtime, "_resolve_device", lambda _value: torch.device("cpu"))
    _, handle = loader(precision="float16")
    assert runtime.open_pipeline(handle) == "pipeline"
    assert captured["model"] == "m-a-p/YuE2-3B"
    assert captured["vae"] == "m-a-p/YuE2-Vae"
    assert captured["torch_dtype"] == torch.float16
    assert captured["cache_dir"] == str(runtime.MODEL_ROOT / "hub")
    assert captured["local_files_only"] is False
    assert captured["verify_hashes"] is True


def test_plan_score_exposes_official_abc_controls():
    plan = SimpleNamespace(
        abc="X:1",
        abc_ids=[1],
        prefix=[2],
        request=SimpleNamespace(to_dict=dict),
        timing={},
        truncated=False,
    )

    class FakePipeline:
        def plan(self, **kwargs):
            self.kwargs = kwargs
            return plan

    handle = runtime.make_config(**loader_values())
    handle._pipeline = FakePipeline()
    result, abc = nodes.YuE2PlanScore.execute(
        handle, "rock", "lyrics", "full", 11, -1.0, "", 0.5, 0.8, 20, 1.01, 75, 16, 100
    ).result
    assert result is plan
    assert abc == "X:1"
    sampling = handle.pipeline().kwargs["abc_sampling"]
    assert (
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.max_tokens,
    ) == (0.5, 0.8, 20, 100)


def test_plan_editor_rebuilds_derived_fields_and_returns_plain_data():
    base = SymbolicPlan(
        SongRequest("rock", "lyrics", cot="full", seed=11),
        "X:1\nK:C\nC|",
        [1],
        [2],
        {"seconds": 1.0},
    )

    class FakePipeline:
        def plan(self, *, request):
            self.request = request
            return SymbolicPlan(
                request,
                request.abc,
                [7, 8],
                [9, 10],
                {"seconds": 0.0},
            )

    handle = runtime.make_config(**loader_values())
    handle._pipeline = FakePipeline()
    edits = json.dumps(
        {
            "request": {"style": "jazz", "seed": 42},
            "abc": "X:2\nK:D\nD|",
            "abc_ids": [999],
            "prefix": [999],
        }
    )
    edited, abc, data = nodes.YuE2PlanEditor.execute(
        handle, base, "X:3\nK:F\nF|", edits
    ).result

    assert edited.request.style == "jazz"
    assert edited.request.lyrics == "lyrics"
    assert edited.request.seed == 42
    assert edited.request.abc == "X:3\nK:F\nF|"
    assert abc == edited.abc == "X:3\nK:F\nF|"
    assert data == {
        "request": edited.request.to_dict(),
        "abc": edited.abc,
        "abc_ids": [7, 8],
        "prefix": [9, 10],
        "timing": {"seconds": 0.0},
        "truncated": False,
    }


def test_plan_editor_clears_score_for_cot_off_and_rejects_bad_edits():
    base = SymbolicPlan(
        SongRequest("rock", "lyrics", cot="full"),
        "X:1\nK:C\nC|",
        [1],
        [2],
    )

    class FakePipeline:
        def plan(self, *, request):
            self.request = request
            return SymbolicPlan(request, None, [], [3])

    handle = runtime.make_config(**loader_values())
    handle._pipeline = FakePipeline()
    edited = nodes.YuE2PlanEditor.execute(handle, base, "", '{"cot":"off"}')[0]
    assert edited.request.cot == "off"
    assert edited.request.abc is None

    with pytest.raises(ValueError, match="received invalid JSON"):
        nodes.YuE2PlanEditor.execute(handle, base, "", "{")
    with pytest.raises(TypeError, match="expects a JSON object"):
        nodes.YuE2PlanEditor.execute(handle, base, "", "[]")
    with pytest.raises(ValueError, match="cannot edit: weights"):
        nodes.YuE2PlanEditor.execute(handle, base, "", '{"weights":"other"}')
    with pytest.raises(ValueError, match="requires nonempty ABC"):
        nodes.YuE2PlanEditor.execute(handle, base, "", '{"abc":null}')


def test_generate_semantic_returns_standard_conditioning_with_semantic_payload():
    semantic = fake_semantic()

    class FakePipeline:
        def generate_semantic(self, plan, **kwargs):
            self.call = (plan, kwargs)
            return semantic

    handle = runtime.make_config(**loader_values())
    handle._pipeline = FakePipeline()
    (conditioning,) = nodes.YuE2GenerateSemantic.execute(
        handle, semantic.plan, 1.0, 0.95, 100, 1.2, 50, 200, 9000
    ).result
    assert conditioning == [[None, {"yue2_semantic": semantic, "yue2_runtime": handle}]]


def test_model_rejects_conditioning_from_another_loader(monkeypatch):
    monkeypatch.setattr(runtime, "_resolve_device", lambda _value: torch.device("cpu"))
    model, _handle = loader(device="cpu")
    _other_model, other_handle = loader(device="cpu")
    with pytest.raises(ValueError, match="same YuE2 Model Loader"):
        model.model.extra_conds(
            yue2_semantic=fake_semantic(), yue2_runtime=other_handle
        )


def test_empty_latent_uses_official_chunk_ranges(monkeypatch):
    semantic = fake_semantic(frames=7)
    conditioning = [[None, {"yue2_semantic": semantic}]]
    monkeypatch.setattr(
        nodes, "chunk_ranges", lambda frames, prefix: [(0, 3), (3, frames)]
    )
    latent = nodes.YuE2EmptyLatent.execute(conditioning)[0]
    chunks = latent["samples"].unbind()
    assert [tuple(chunk.shape) for chunk in chunks] == [(1, 64, 3), (1, 64, 4)]
    assert all(torch.count_nonzero(chunk) == 0 for chunk in chunks)


def test_empty_latent_requires_yue2_conditioning():
    with pytest.raises(ValueError, match="YuE2 Generate Semantic"):
        nodes.YuE2EmptyLatent.execute([[None, {}]])


def test_random_noise_layout_is_remapped_to_official_token_major_draw():
    generator = torch.Generator(device="cpu").manual_seed(123)
    comfy_noise = torch.randn((1, 64, 5), generator=generator)
    official = torch.randn(
        (5, 64), generator=torch.Generator(device="cpu").manual_seed(123)
    )
    remapped = nodes._official_noise_layout(comfy_noise)
    assert torch.equal(remapped[0].T, official)


def test_explicit_midpoint_matches_reference_and_runs_chunk_major():
    sigmas = torch.linspace(1, 0, 9, dtype=torch.float64)
    chunks = [torch.full((1, 64, 2), 0.1), torch.full((1, 64, 3), -0.2)]
    calls = []

    def velocity(state, sigma, chunk_index):
        calls.append((chunk_index, float(sigma)))
        return state.square() + sigma

    actual = nodes._explicit_midpoint(chunks, sigmas, velocity)
    expected = []
    for source in chunks:
        state = source.clone()
        for start, end in pairwise(sigmas):
            dt = start - end
            first = state.square() + start
            mid = state - first * (dt / 2)
            state = state - (mid.square() + (start + end) / 2) * dt
        expected.append(state)
    assert all(
        torch.allclose(a, b, rtol=0, atol=1e-7) for a, b in zip(actual, expected)
    )
    assert [index for index, _ in calls] == [0] * 16 + [1] * 16


def test_linear_schedule_is_exact_official_grid():
    sigmas = nodes.YuE2LinearSchedule.execute(32)[0]
    assert sigmas.dtype == torch.float64
    assert torch.equal(sigmas, torch.linspace(1, 0, 33, dtype=torch.float64))


def test_native_random_noise_basic_guider_and_sampler_custom_chain(monkeypatch):
    import yue2.nar
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        RandomNoise,
        SamplerCustomAdvanced,
    )

    monkeypatch.setattr(runtime, "_resolve_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(
        nodes, "chunk_ranges", lambda frames, _prefix: [(0, 2), (2, frames)]
    )
    model, handle = loader(precision="float32", device="cpu")
    semantic = fake_semantic(frames=5, prefix_tokens=2, seed=123)

    nar_model = SimpleNamespace(vae2llm=torch.nn.Linear(1, 1, bias=False))

    class FakePipeline:
        quantization = "none"
        _model = nar_model
        device = torch.device("cpu")

        def _load_model(self, for_nar=False):
            assert for_nar
            return nar_model

    class FakeEngine:
        def __init__(self, _model, chunk):
            self.chunk = chunk

        def velocity(self, state, _raw_t):
            return torch.zeros_like(state)

        def close(self):
            pass

    handle._pipeline = FakePipeline()
    monkeypatch.setattr(yue2.nar, "CachedNAR", FakeEngine)
    conditioning = [[None, {"yue2_semantic": semantic, "yue2_runtime": handle}]]
    latent = nodes.YuE2EmptyLatent.execute(conditioning)[0]
    noise = RandomNoise.execute(123)[0]
    guider = BasicGuider.execute(model, conditioning)[0]
    sampler = nodes.YuE2ExplicitMidpoint.execute()[0]
    sigmas = nodes.YuE2LinearSchedule.execute(2)[0]
    sampled = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0][
        "samples"
    ]
    if getattr(sampled, "is_nested", False):
        sampled = torch.cat(sampled.unbind(), dim=-1)
    actual = sampled[0].T
    expected = torch.randn(
        (5, 64), generator=torch.Generator(device="cpu").manual_seed(123)
    )
    assert torch.equal(actual, expected)


def test_decode_accepts_nested_standard_latent():
    class FakePipeline:
        def decode_audio(self, latents, **kwargs):
            self.call = (latents, kwargs)
            return torch.zeros((1, 2, 64))

    handle = runtime.make_config(**loader_values())
    handle._pipeline = FakePipeline()
    nested = nodes.comfy.nested_tensor.NestedTensor(
        [torch.ones((1, 64, 2)), torch.ones((1, 64, 3)) * 2]
    )
    audio = nodes.YuE2Decode.execute(handle, {"samples": nested}, False)[0]
    assert handle.pipeline().call[0].shape == (1, 64, 5)
    assert audio["waveform"].shape == (1, 2, 64)
    assert audio["sample_rate"] == 48000


def test_handle_opens_pipeline_once(monkeypatch):
    opened = []
    pipe = object()
    monkeypatch.setattr(
        runtime, "open_pipeline", lambda config: opened.append(config) or pipe
    )
    handle = runtime.make_config(**loader_values())
    assert opened == []
    assert handle.pipeline() is pipe
    assert handle.pipeline() is pipe
    assert opened == [handle.config]


def test_pipeline_loads_main_model_through_comfy_model_management(
    monkeypatch, tmp_path
):
    pipeline_type = runtime._pipeline_type()
    pipe = pipeline_type.__new__(pipeline_type)
    pipe._model = None
    pipe._model_patcher = None
    pipe.model_dir = tmp_path
    pipe.torch_dtype = torch.float32
    pipe.offload_device = torch.device("cpu")
    pipe.device = torch.device("cpu")
    pipe.quantization = "none"
    pipe.load_timing = {}
    pipe._status = lambda *_args, **_kwargs: nullcontext()

    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        yue2.modeling_yue2.YuE2ForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    loaded = []
    monkeypatch.setattr(
        runtime.model_management,
        "load_models_gpu",
        lambda patchers, **kwargs: loaded.append((patchers, kwargs)),
    )
    monkeypatch.setattr(
        runtime.model_management, "archive_model_dtypes", lambda _model: None
    )

    assert pipe._load_model() is model
    assert pipe._model_patcher.model.module is model
    assert loaded == [([pipe._model_patcher], {"force_full_load": True})]


def test_decoder_offloads_main_model_and_uses_comfy_model_management(
    monkeypatch, tmp_path
):
    import yue2.modeling_vae

    pipeline_type = runtime._pipeline_type()
    pipe = pipeline_type.__new__(pipeline_type)
    pipe._vae = None
    pipe._vae_patcher = None
    pipe._model_patcher = object()
    pipe.vae_dir = tmp_path
    pipe.offload_ar = True
    pipe.offload_device = torch.device("cpu")
    pipe.device = torch.device("cpu")
    pipe._status = lambda *_args, **_kwargs: nullcontext()

    vae = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        yue2.modeling_vae.YuE2VAE, "from_pretrained", lambda *_args, **_kwargs: vae
    )
    unloaded = []
    loaded = []
    monkeypatch.setattr(
        runtime.model_management,
        "unload_model_and_clones",
        lambda patcher: unloaded.append(patcher),
    )
    monkeypatch.setattr(
        runtime.model_management,
        "load_models_gpu",
        lambda patchers, **kwargs: loaded.append((patchers, kwargs)),
    )
    monkeypatch.setattr(
        runtime.model_management, "archive_model_dtypes", lambda _model: None
    )

    assert pipe._load_vae() is vae
    assert unloaded == [pipe._model_patcher]
    assert pipe._vae_patcher.model.module is vae
    assert loaded == [([pipe._vae_patcher], {"force_full_load": True})]
