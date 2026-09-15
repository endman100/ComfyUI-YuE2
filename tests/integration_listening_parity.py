from __future__ import annotations

import argparse
import json
from pathlib import Path

from integration_parity import COMFY_ROOT, nodes, run_case
from yue2.protocol import GenerationConfig


CITY_LIGHTS_LYRICS = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
Fold the night and leave it here
Morning has a sky to clear

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night"""

CITY_LIGHTS_MELODY = """X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
E2G2A2G2E2D2C4|D2E2G2E2D2C2D4|E2G2A2c2B2A2G4|F2E2D2E2G2E2C4|
V: Ins
Z4|
% chorus
V: Vocal
G2A2c2B2A2G2E4|F2A2G2E2D2E2G4|A2c2B2A2G2E2D4|E2G2A2G2E2D2C4|
V: Ins
Z4|
"""

DEFAULTS = GenerationConfig()
CASES = [
    {
        "name": "01_official_city_lights_default",
        "style": (
            "English, warm piano pop, expressive female voice, acoustic piano, "
            "rounded bass and light drums, lyrical memorable melody, unhurried "
            "phrasing, 88 BPM"
        ),
        "lyrics": CITY_LIGHTS_LYRICS,
        "cot": "full",
        "seed": 831001,
        "abc": "",
        "abc_sampling": DEFAULTS.abc,
        "semantic": DEFAULTS.semantic,
        "full_decode": False,
    },
    {
        "name": "02_electronic_instrumental_default",
        "style": (
            "Instrumental progressive electronic, analog synthesizers, deep bass, "
            "crisp drums, evolving arpeggios, spacious breakdown, 118 BPM"
        ),
        "lyrics": "[Instrumental]",
        "cot": "off",
        "seed": 2026,
        "abc": "",
        "semantic": DEFAULTS.semantic,
        "full_decode": False,
    },
    {
        "name": "03_rock_vocal_default",
        "style": (
            "English energetic alternative rock, gritty male vocal, electric "
            "guitars, live drums, anthemic chorus, 132 BPM"
        ),
        "lyrics": (
            "[Verse]\n"
            "The road keeps calling my name\n"
            "Streetlights flicker into flame\n"
            "Every mile is pulling near\n"
            "I can hear the future clear\n\n"
            "[Chorus]\n"
            "We rise again, we won't let go\n"
            "Through the night our colors glow\n"
            "Raise your voice and cross the line\n"
            "This restless heart is keeping time"
        ),
        "cot": "off",
        "seed": 987654,
        "abc": "",
        "semantic": DEFAULTS.semantic,
        "full_decode": False,
    },
    {
        "name": "04_auto_abc_full_default",
        "style": (
            "English cinematic folk ballad, intimate female vocal, acoustic "
            "guitar, cello, brushed drums, warm gradual build, 84 BPM"
        ),
        "lyrics": (
            "[Verse]\n"
            "Carry the morning home\n"
            "Over the fields we used to roam\n"
            "Gather the letters by the door\n"
            "Read every promise made before\n\n"
            "[Chorus]\n"
            "Stay where the quiet river bends\n"
            "Sing till the winter finally ends\n"
            "Hold to the light we used to know\n"
            "Carry the morning as we go"
        ),
        "cot": "full",
        "seed": 42,
        "abc": "",
        "abc_sampling": DEFAULTS.abc,
        "semantic": DEFAULTS.semantic,
        "full_decode": False,
    },
    {
        "name": "05_supplied_abc_melody_default",
        "style": (
            "English bright chamber pop rearrangement, duet vocals, strings, "
            "piano, rounded bass, light drums, 88 BPM"
        ),
        "lyrics": CITY_LIGHTS_LYRICS,
        "cot": "melody",
        "seed": 314159,
        "abc": CITY_LIGHTS_MELODY,
        "semantic": DEFAULTS.semantic,
        "full_decode": True,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMFY_ROOT / "output" / "yue2_listening_parity_5cases",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, runtime = nodes.YuE2ModelLoader.execute(
        "hugging_face",
        "m-a-p/YuE2-3B",
        "m-a-p/YuE2-Vae",
        "bfloat16",
        "cuda",
        24.0,
        False,
        True,
        True,
        "",
        "",
        "",
    ).result
    results = []
    try:
        for case in CASES:
            results.append(run_case(model, runtime, case, args.output_dir))
    finally:
        if runtime._pipeline is not None:
            runtime._pipeline.close()

    summary = {
        "comparison": "default-length Official YuE2 versus ComfyUI listening run",
        "semantic_sampling": {
            "min_tokens": DEFAULTS.semantic.min_tokens,
            "max_tokens": DEFAULTS.semantic.max_tokens,
        },
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary["passed"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
