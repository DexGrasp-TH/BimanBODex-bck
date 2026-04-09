#!/usr/bin/env python3

import argparse
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIP_ROOT = REPO_ROOT / "src/curobo/content/configs/manip"
ASSET_ROOT = REPO_ROOT / "src/curobo/content/assets"
OUTPUT_ROOT = ASSET_ROOT / "output"

CONFIGS_BY_HAND = {
    "shadow": [
        "sim_shadow/tabletop_two",
        "sim_shadow/tabletop_three",
        "sim_shadow/tabletop_full",
        "sim_dual_dummy_arm_shadow/tabletop_three",
        "sim_dual_dummy_arm_shadow/tabletop_full",
    ],
    "leap": [
        "sim_leap/tabletop_two",
        "sim_leap/tabletop_three",
        "sim_leap/tabletop_full",
        "sim_dual_dummy_arm_leap/tabletop_three",
        "sim_dual_dummy_arm_leap/tabletop_full",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze grasp counts produced by scripts/run_all_grasps_multi_gpu.sh."
    )
    parser.add_argument("--hand", choices=sorted(CONFIGS_BY_HAND), required=True)
    parser.add_argument("--exp-name", required=True, help="Experiment name passed to run_all_grasps_multi_gpu.sh")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Override world.start for analysis, matching the run script.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Override world.end for analysis, matching the run script.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=0,
        help="Show up to N missing output file paths per config.",
    )
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_expected_inputs(config_data, cli_start, cli_end):
    world_cfg = config_data["world"]
    template_path = ASSET_ROOT / world_cfg["template_path"]
    all_paths = sorted(glob(str(template_path)))

    object_scale_list = world_cfg.get("object_scale_list")
    if object_scale_list is not None:
        scale_patterns = [f"scale{int(scale * 100):03d}_" for scale in object_scale_list]
        all_paths = [path for path in all_paths if any(pattern in Path(path).name for pattern in scale_patterns)]

    start = world_cfg.get("start") if cli_start is None else cli_start
    end = world_cfg.get("end") if cli_end is None else cli_end

    start = 0 if start is None else start
    return [Path(path) for path in all_paths[start:end]]


def expected_output_path(config_path: str, exp_name: str, input_path: Path) -> Path:
    scene_root = ASSET_ROOT / "object/DGN_2k/scene_cfg"
    rel_path = input_path.relative_to(scene_root)
    return OUTPUT_ROOT / config_path / exp_name / "graspdata" / rel_path.parent / f"{input_path.stem}_grasp.npy"


def analyze_config(config_path: str, exp_name: str, cli_start, cli_end):
    config_yaml = MANIP_ROOT / f"{config_path}.yml"
    config_data = load_yaml(config_yaml)
    expected_inputs = get_expected_inputs(config_data, cli_start, cli_end)

    expected_outputs = [expected_output_path(config_path, exp_name, input_path) for input_path in expected_inputs]
    existing_outputs = [path for path in expected_outputs if path.is_file()]
    missing_outputs = [path for path in expected_outputs if not path.is_file()]

    return {
        "config_path": config_path,
        "expected": len(expected_outputs),
        "done": len(existing_outputs),
        "missing": len(missing_outputs),
        "missing_outputs": missing_outputs,
    }


def main():
    args = parse_args()
    rows = [
        analyze_config(config_path, args.exp_name, args.start, args.end)
        for config_path in CONFIGS_BY_HAND[args.hand]
    ]

    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hand: {args.hand}")
    print(f"Experiment: {args.exp_name}")
    if args.start is not None or args.end is not None:
        print(f"World range override: start={args.start if args.start is not None else '<config>'}, end={args.end if args.end is not None else '<config>'}")
    print("")
    print(
        f"{'Config':45} {'Expected':>8} {'Done':>8} {'Missing':>8} {'Percent':>8}"
    )
    print("-" * 82)

    total_expected = 0
    total_done = 0
    total_missing = 0

    for row in rows:
        percent = (100.0 * row["done"] / row["expected"]) if row["expected"] else 0.0
        print(
            f"{row['config_path']:45} {row['expected']:8d} {row['done']:8d} {row['missing']:8d} {percent:7.2f}%"
        )
        total_expected += row["expected"]
        total_done += row["done"]
        total_missing += row["missing"]

    total_percent = (100.0 * total_done / total_expected) if total_expected else 0.0
    print("-" * 82)
    print(
        f"{'TOTAL':45} {total_expected:8d} {total_done:8d} {total_missing:8d} {total_percent:7.2f}%"
    )

    if args.show_missing > 0:
        for row in rows:
            if not row["missing_outputs"]:
                continue
            print("")
            print(f"Missing examples for {row['config_path']}:")
            for path in row["missing_outputs"][: args.show_missing]:
                print(path)


if __name__ == "__main__":
    main()
