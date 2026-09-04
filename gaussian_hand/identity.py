"""Identity boundary for subject-specific Gaussian Hand assets."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SUBJECT_RE = re.compile(r"^P\d{4}$")


@dataclass(frozen=True)
class SubjectBinding:
    """The immutable dataset/subject/hand identity of one Gaussian model."""

    dataset: str
    subject_id: str
    handedness: str
    model_id: str
    sequences: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubjectBinding":
        binding = value.get("binding", value)
        item = cls(
            dataset=str(binding["dataset"]),
            subject_id=str(binding["subject_id"]),
            handedness=str(binding["handedness"]).lower(),
            model_id=str(binding["model_id"]),
            sequences=tuple(str(x) for x in value.get("sequences", ())),
        )
        item.validate()
        return item

    @classmethod
    def from_json(cls, path: str | Path) -> "SubjectBinding":
        with Path(path).open() as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> None:
        if self.dataset != "HOT3D":
            raise ValueError(f"Unsupported Gaussian Hand dataset: {self.dataset!r}")
        if not _SUBJECT_RE.fullmatch(self.subject_id):
            raise ValueError(f"Invalid HOT3D subject id: {self.subject_id!r}")
        if self.handedness not in {"left", "right"}:
            raise ValueError("handedness must be 'left' or 'right'")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.sequences:
            raise ValueError("at least one source sequence is required")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("source sequences must be unique")
        prefix = self.subject_id + "_"
        foreign = [name for name in self.sequences if not name.startswith(prefix)]
        if foreign:
            raise ValueError(
                f"model {self.model_id!r} is bound to {self.subject_id}; "
                f"foreign sequences are forbidden: {foreign}"
            )

    def assert_compatible(self, metadata: Mapping[str, Any]) -> None:
        """Reject a checkpoint/model card belonging to another hand identity."""
        actual = (metadata.get("dataset"), metadata.get("subject_id"),
                  metadata.get("handedness"))
        expected = (self.dataset, self.subject_id, self.handedness)
        if actual != expected:
            raise ValueError(f"Gaussian Hand identity mismatch: {actual} != {expected}")
