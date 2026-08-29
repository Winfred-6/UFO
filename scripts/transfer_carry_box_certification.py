"""Transfer physics-certified reset masks between byte-equivalent carry clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl


def _identity(record: dict) -> tuple[str, str, int | None]:
    metadata = record.get("metadata") or {}
    return (
        str(record.get("base_sequence_id", metadata.get("base_sequence_id", ""))),
        str(record.get("augmentation", metadata.get("augmentation", "original"))),
        metadata.get("augmentation_index"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    certified_path = args.certified.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    certified = joblib.load(certified_path)
    target = joblib.load(target_path)
    source_by_identity = {_identity(record): (key, record) for key, record in certified.items()}
    if len(source_by_identity) != len(certified):
        raise ValueError("Certified dataset contains duplicate base/augmentation identities")

    compare_fields = (
        "root_trans_offset",
        "root_quat",
        "dof_pos",
        "object_pos",
        "object_quat",
        "object_lin_vel",
        "object_ang_vel",
        "object_goal_pos",
        "object_phase",
    )
    copied_frames = 0
    for target_key, target_record in target.items():
        identity = _identity(target_record)
        if identity not in source_by_identity:
            raise KeyError(f"No certified source trajectory matches target={target_key!r} identity={identity!r}")
        source_key, source_record = source_by_identity[identity]
        for field in compare_fields:
            source_value = np.asarray(source_record[field])
            target_value = np.asarray(target_record[field])
            if source_value.shape != target_value.shape or not np.allclose(
                source_value,
                target_value,
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Certification transfer mismatch target={target_key!r} source={source_key!r} field={field!r}"
                )
        reset_mask = np.asarray(source_record["object_stage_reset_valid"], dtype=np.float32).copy()
        target_record["object_stage_reset_valid"] = reset_mask
        metadata = dict(target_record.get("metadata") or {})
        metadata["object_stage_reset_safety"] = dict(
            (source_record.get("metadata") or {})["object_stage_reset_safety"]
        )
        metadata["object_stage_reset_safety"]["transferred_from"] = source_key
        target_record["metadata"] = metadata
        copied_frames += int(np.count_nonzero(reset_mask > 0.5))

    dump_ufo_pkl(target, output_path, "carry_box_certification_transfer")
    print(
        {
            "certified": str(certified_path),
            "target": str(target_path),
            "output": str(output_path),
            "motions": len(target),
            "certified_frames": copied_frames,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
