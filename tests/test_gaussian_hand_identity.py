import json

import pytest

from gaussian_hand import SubjectBinding


def test_p0001_config_is_single_subject_bound() -> None:
    binding = SubjectBinding.from_json(
        "gaussian_hand/configs/subjects/P0001_right.json"
    )
    assert binding.subject_id == "P0001"
    assert binding.handedness == "right"
    assert all(x.startswith("P0001_") for x in binding.sequences)


def test_foreign_subject_sequence_is_rejected() -> None:
    config = json.loads(
        open("gaussian_hand/configs/subjects/P0001_right.json").read()
    )
    config["sequences"].append("P0009_not_allowed")
    with pytest.raises(ValueError, match="foreign sequences"):
        SubjectBinding.from_dict(config)


def test_checkpoint_identity_mismatch_is_rejected() -> None:
    binding = SubjectBinding.from_json(
        "gaussian_hand/configs/subjects/P0001_right.json"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        binding.assert_compatible(
            {"dataset": "HOT3D", "subject_id": "P0001", "handedness": "left"}
        )
