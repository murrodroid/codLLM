"""Fast unit tests for select_floor_upsample_targets, especially the
singlecod_only filter that prevents compound multi-CoD labels from being
treated as rare classes and amplified by floor upsampling."""

from __future__ import annotations

import pandas as pd
import pytest

from codllm.data.balancing import select_floor_upsample_targets


def _frame(labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"label": labels})


class TestSelectFloorUpsampleTargets:
    def test_floor_zero_returns_empty(self) -> None:
        df = _frame(["A", "A", "B"])
        assert select_floor_upsample_targets(df, "label", floor=0) == {}

    def test_default_upsamples_rare_classes(self) -> None:
        # A appears 1x, B 3x, C 5x; floor=4 upsamples A and B to 4.
        df = _frame(["A"] + ["B"] * 3 + ["C"] * 5)
        targets = select_floor_upsample_targets(df, "label", floor=4, decay=0.0)
        assert targets == {"A": 4, "B": 4}
        assert "C" not in targets  # already at/above floor

    def test_default_treats_compound_labels_as_rare_classes(self) -> None:
        """Default behavior (no singlecod_only): compound multi-CoD labels
        get upsampled just like single-CoD labels. This is the legacy behavior
        and the cause of the train/eval distribution shift bug."""
        labels = ["A"] * 60 + ["B,C"] * 2 + ["D,E,F"]
        df = _frame(labels)
        targets = select_floor_upsample_targets(df, "label", floor=10, decay=0.0)
        # A is above floor so not in targets; B,C and D,E,F are rare and get
        # bumped because the default treats them as classes.
        assert "A" not in targets
        assert "B,C" in targets
        assert "D,E,F" in targets

    def test_singlecod_only_excludes_compound_labels(self) -> None:
        """With singlecod_only=True, compound labels are filtered out so
        the floor only protects rare ICD10h *codes*, not rare *combinations*."""
        labels = ["A"] * 2 + ["B"] * 60 + ["C,D"] * 2 + ["E,F,G"]
        df = _frame(labels)
        targets = select_floor_upsample_targets(
            df,
            "label",
            floor=10,
            decay=0.0,
            singlecod_only=True,
        )
        # A is a rare single-CoD class -> still upsampled.
        assert "A" in targets
        # B is common single-CoD -> not upsampled.
        assert "B" not in targets
        # Compound labels are EXCLUDED entirely under singlecod_only.
        assert "C,D" not in targets
        assert "E,F,G" not in targets

    def test_singlecod_only_respects_custom_separator(self) -> None:
        labels = ["A"] * 2 + ["B|C"] * 2
        df = _frame(labels)
        targets_default_sep = select_floor_upsample_targets(
            df, "label", floor=5, decay=0.0, singlecod_only=True
        )
        # Default separator is ',' so 'B|C' is treated as single-CoD.
        assert "B|C" in targets_default_sep

        targets_pipe_sep = select_floor_upsample_targets(
            df,
            "label",
            floor=5,
            decay=0.0,
            singlecod_only=True,
            label_separator="|",
        )
        # With separator='|' the compound label is now correctly excluded.
        assert "B|C" not in targets_pipe_sep
        assert "A" in targets_pipe_sep

    def test_singlecod_only_off_by_default(self) -> None:
        """Backward compatibility: existing callers without the new flag
        keep the legacy compound-label-amplifying behavior."""
        labels = ["A"] + ["B,C"]
        df = _frame(labels)
        targets = select_floor_upsample_targets(df, "label", floor=5, decay=0.0)
        assert "B,C" in targets  # legacy: compounds counted

    def test_rejects_negative_floor(self) -> None:
        with pytest.raises(ValueError, match="floor must be non-negative"):
            select_floor_upsample_targets(_frame(["A"]), "label", floor=-1)

    def test_rejects_invalid_decay(self) -> None:
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            select_floor_upsample_targets(
                _frame(["A"]), "label", floor=5, decay=1.5
            )
