from pathlib import Path
import threading
import time

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import codllm.data_handler as data_handler_module
from codllm.config import Config, DataSourceConfig
from codllm.dataset_input import COPENHAGEN_MAPPING
from codllm.data_handler import (
    DataHandler,
    DatasetMapping,
    MAPPING_REGISTRY,
    _build_text,
    _build_y,
    build_and_save_processed_dataset,
    build_processed_dataset,
    load_dataset,
    load_source_dataset,
    manipulate_classes,
    select_upsample_targets,
    upsample,
)


def _make_mapping(**overrides) -> DatasetMapping:
    """Create a mapping fixture with compact column defaults."""
    defaults = dict(
        text_col=0,
        single_code_col=1,
        multi_code_cols=[2],
        sex_col=3,
        sex_map={"1": "male", "2": "female"},
        age_col=4,
        record_id_col=5,
    )
    defaults.update(overrides)
    return DatasetMapping(**defaults)


def _row(*values) -> pd.Series:
    """Build one positional row for mapping unit tests."""
    return pd.Series(values)


def _sample_df() -> pd.DataFrame:
    """Build a small dataframe with cod, labels, sex, age, and record id."""
    return pd.DataFrame(
        [
            ["cholera", "A00", "", "1", "2.4", "RID-001"],
            ["typhus", "A01", "J18", "2", "40", "RID-002"],
        ]
    )


def _copenhagen_row(
    record_id: str,
    cod_text: str,
    icd10h_code: str | None,
    sex: str = "Mand",
    age: str = "35",
) -> list[str | None]:
    """Build one Copenhagen-style row with required positional columns populated."""
    row: list[str | None] = [""] * 40
    row[0] = record_id
    row[12] = age
    row[23] = sex
    row[37] = cod_text
    row[39] = icd10h_code
    return row


def _processed_df(num_rows: int) -> pd.DataFrame:
    """Build a processed dataframe fixture with required training columns."""
    return pd.DataFrame(
        {
            "source_id": ["test_source"] * num_rows,
            "record_id": [f"RID-{idx:03d}" for idx in range(num_rows)],
            "source_path": ["sample.csv"] * num_rows,
            "text": [
                f"cod: text-{idx} | age: {idx} | sex: male" for idx in range(num_rows)
            ],
            "y_codes": [["A00"] for _ in range(num_rows)],
            "label": [f"A{idx:03d}" for idx in range(num_rows)],
        }
    )


def _balance_df() -> pd.DataFrame:
    """Build a small class-imbalanced dataframe for balancing tests."""
    return pd.DataFrame(
        {
            "text": [
                "cod: alpha | age: 1 | sex: male",
                "cod: beta | age: 2 | sex: male",
                "cod: gamma | age: 3 | sex: female",
                "cod: delta | age: 4 | sex: female",
            ],
            "label": ["A00", "A00", "B00", "C00"],
        }
    )


def _write_masterlist(path: Path, num_rows: int = 4) -> None:
    """Write a compact ICD10h masterlist-like workbook for pretraining tests."""
    rows = []
    for idx in range(num_rows):
        rows.append(
            {
                "IDMasterlist": idx + 1,
                "ICD10h": f"A{idx:02d}.000",
                "ICD10": f"A{idx:02d}.0",
                "icd10_2levelCATEGORY": "Category",
                "ICD10_2levelCAUSE": f"Cause {idx}",
                "ICD10h_DESCRIPTION": f"description-{idx}",
                "HistCat": "Hist",
                "DoNotUse": 0,
                "NotForUnderlying": 0,
                "GenderSpecific": 0,
            }
        )
    pd.DataFrame(rows).to_excel(path, sheet_name="Masterlist", index=False)


class TestBuildText:
    def test_build_text_uses_configured_training_input_order(self) -> None:
        """Text should respect feature order from training_input."""
        mapping = _make_mapping()
        row = _row("cholera", "A00", "", "1", "2.4", "RID-001")
        result = _build_text(row, mapping, ["cod", "age", "sex"])
        assert result == "cod: cholera | age: 2.4 | sex: male"

    def test_build_text_handles_missing_values(self) -> None:
        """Missing values should resolve to unknown markers."""
        mapping = _make_mapping()
        row = _row(None, "A00", "", None, None, "RID-001")
        result = _build_text(row, mapping, ["cod", "age", "sex"])
        assert result == "cod: unknown | age: unknown | sex: unknown"

    def test_build_text_uses_configured_field_separator(self) -> None:
        """Text should use the configured separator between input fields."""
        mapping = _make_mapping()
        row = _row("cholera", "A00", "", "1", "2.4", "RID-001")
        result = _build_text(
            row,
            mapping,
            ["cod", "age", "sex"],
            field_separator=" || ",
        )
        assert result == "cod: cholera || age: 2.4 || sex: male"


class TestBuildY:
    def test_build_y_collects_all_available_labels(self) -> None:
        """y should preserve all collected labels from multi-code columns."""
        mapping = _make_mapping(multi_code_cols=[2, 6])
        row = _row("text", "A00", "J18", "1", "20", "RID-001", "R99")
        assert _build_y(row, mapping) == ["J18", "R99"]

    def test_build_y_falls_back_to_single_code_when_multicode_is_empty(self) -> None:
        """y should use the single-code field when no multicode labels are present."""
        mapping = _make_mapping(multi_code_cols=[2, 6])
        row = _row("text", "A00", "", "1", "20", "RID-001", "")
        assert _build_y(row, mapping) == ["A00"]


class TestLoaders:
    def test_load_dataset_reads_semicolon_csv(self, tmp_path: Path) -> None:
        """Legacy loader should support semicolon-separated CSV files."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False, sep=";")
        mapping = _make_mapping()
        result = load_dataset(
            str(csv_path), mapping, training_input=["cod", "age", "sex"], sep=";"
        )
        assert list(result.columns) == ["text", "y"]
        assert result.iloc[0]["text"] == "cod: cholera | age: 2.4 | sex: male"
        assert result.iloc[0]["y"] == ["A00"]
        assert result.iloc[1]["y"] == ["J18"]

    def test_load_source_dataset_emits_canonical_columns(self, tmp_path: Path) -> None:
        """Source loader should create canonical processed columns."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        source = DataSourceConfig(
            source_id="test_source", path=str(csv_path), mapping_id="test_mapping"
        )
        mapping = _make_mapping()
        result = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=["cod", "age", "sex"],
            max_labels=1,
            data_raw_dir="",
        )
        assert "source_id" in result.columns
        assert "record_id" in result.columns
        assert "source_path" in result.columns
        assert "text" in result.columns
        assert "y_codes" in result.columns
        assert "label" in result.columns
        assert result.iloc[0]["source_id"] == "test_source"
        assert result.iloc[0]["record_id"] == "RID-001"
        assert result.iloc[0]["label"] == "A00"

    def test_load_source_dataset_drops_nan_like_codes_before_dataset_assembly(
        self, tmp_path: Path
    ) -> None:
        """Rows with NaN-like label placeholders should be removed before assembly."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                ["cholera", "nan", "", "1", "2.4", "RID-001"],
                ["typhus", "A01", "", "2", "40", "RID-002"],
            ]
        ).to_csv(csv_path, index=False)
        source = DataSourceConfig(
            source_id="test_source", path=str(csv_path), mapping_id="test_mapping"
        )
        mapping = _make_mapping(multi_code_cols=[])
        result = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=["cod", "age", "sex"],
            max_labels=1,
            data_raw_dir="",
        )
        assert len(result) == 1
        assert result.iloc[0]["record_id"] == "RID-002"
        assert result.iloc[0]["label"] == "A01"

    def test_build_processed_dataset_supports_copenhagen_mapping(
        self, tmp_path: Path
    ) -> None:
        """Copenhagen mapping should load and drop rows without valid ICD10h codes."""
        csv_path = tmp_path / "copenhagen.csv"
        pd.DataFrame(
            [
                _copenhagen_row("CPH-001", "tekst-1", None),
                _copenhagen_row("CPH-002", "tekst-2", "A00.000"),
                _copenhagen_row("CPH-003", "tekst-3", "B01.001"),
            ]
        ).to_csv(csv_path, index=False)

        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="copenhagen_source",
                    path="copenhagen.csv",
                    mapping_id="copenhagen",
                )
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=1,
        )
        assert "copenhagen" in MAPPING_REGISTRY
        assert COPENHAGEN_MAPPING.multi_code_cols == []

        result = build_processed_dataset(cfg)
        assert len(result) == 2
        assert result["record_id"].tolist() == ["CPH-002", "CPH-003"]
        assert result["label"].tolist() == ["A00.000", "B01.001"]

    def test_build_processed_dataset_combines_sources(self, tmp_path: Path) -> None:
        """Processed dataset should concatenate all configured sources."""
        csv_path = tmp_path / "source_one.csv"
        xlsx_path = tmp_path / "source_two.xlsx"
        _sample_df().to_csv(csv_path, index=False)
        _sample_df().to_excel(xlsx_path, index=False)

        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="source_one.csv",
                    mapping_id="test_mapping",
                ),
                DataSourceConfig(
                    source_id="xlsx_source",
                    path="source_two.xlsx",
                    mapping_id="test_mapping",
                ),
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=1,
        )
        mapping = _make_mapping()
        result = build_processed_dataset(
            cfg, mapping_registry={"test_mapping": mapping}
        )
        assert len(result) == 4
        assert sorted(result["source_id"].unique()) == ["csv_source", "xlsx_source"]
        assert set(result["label"]) == {"A00", "J18"}

    def test_build_processed_dataset_uses_configured_text_and_label_columns(
        self, tmp_path: Path
    ) -> None:
        """Processed output should honor configured text/label column names."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="sample.csv",
                    mapping_id="test_mapping",
                )
            ],
            dataset_text_column="prompt",
            dataset_label_column="target",
            training_input=["cod", "age", "sex"],
            max_label_count=1,
        )
        mapping = _make_mapping()
        result = build_processed_dataset(
            cfg, mapping_registry={"test_mapping": mapping}
        )
        assert "prompt" in result.columns
        assert "target" in result.columns
        assert "text" not in result.columns
        assert "label" not in result.columns
        assert result.iloc[0]["target"] == "A00"

    def test_build_processed_dataset_rejects_unknown_training_input(
        self, tmp_path: Path
    ) -> None:
        """Unsupported training_input values should fail fast."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source", path="sample.csv", mapping_id="test_mapping"
                )
            ],
            training_input=["cod", "city"],
        )
        mapping = _make_mapping()
        with pytest.raises(ValueError):
            build_processed_dataset(cfg, mapping_registry={"test_mapping": mapping})

    def test_build_processed_dataset_uses_configured_label_separator(
        self, tmp_path: Path
    ) -> None:
        """Processed labels should follow the configured separator."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="sample.csv",
                    mapping_id="test_mapping",
                )
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=2,
            label_separator=",",
        )
        mapping = _make_mapping(multi_code_cols=[2, 6])
        raw = pd.DataFrame(
            [["text", "A00.000", "J18.100", "1", "20", "RID-001", "R99.900"]]
        )
        raw.to_csv(csv_path, index=False)
        result = build_processed_dataset(
            cfg, mapping_registry={"test_mapping": mapping}
        )
        assert result.iloc[0]["label"] == "J18.100,R99.900"

    def test_load_source_dataset_excludes_rows_above_max_label_count(
        self, tmp_path: Path
    ) -> None:
        """Rows with more than max_labels should be removed from processed output."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                ["text-one", "A00.000", "J18.100", "1", "20", "RID-001", "R99.900"],
                ["text-two", "B01.001", "B01.001", "1", "21", "RID-002", ""],
            ]
        ).to_csv(csv_path, index=False)

        source = DataSourceConfig(
            source_id="test_source", path=str(csv_path), mapping_id="test_mapping"
        )
        mapping = _make_mapping(multi_code_cols=[2, 6])
        result = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=["cod", "age", "sex"],
            max_labels=1,
            data_raw_dir="",
        )
        assert len(result) == 1
        assert result.iloc[0]["record_id"] == "RID-002"
        assert result.iloc[0]["label"] == "B01.001"

    def test_build_and_save_processed_dataset_writes_output_file(
        self, tmp_path: Path
    ) -> None:
        """High-level builder should persist processed data to configured output path."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        output_dir = tmp_path / "processed"
        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(output_dir),
            processed_filename="training.csv",
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source", path="sample.csv", mapping_id="test_mapping"
                )
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=1,
        )
        mapping = _make_mapping()
        result = build_and_save_processed_dataset(
            cfg, mapping_registry={"test_mapping": mapping}
        )
        output_path = output_dir / "training.csv"
        assert output_path.exists()
        persisted = pd.read_csv(output_path)
        assert len(persisted) == len(result)


class TestDataHandler:
    def test_get_pretraining_train_dataframe_loads_masterlist(
        self, tmp_path: Path
    ) -> None:
        """Pretraining loader should map ICD10h description/text into training columns."""
        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist(masterlist_path, num_rows=3)

        cfg = Config(
            pretrain_enabled=True,
            pretrain_masterlist_path=str(masterlist_path),
            pretrain_masterlist_sheet_name="Masterlist",
            pretrain_dataset_size=1.0,
            training_input=["cod"],
            data_sources=[],
        )
        handler = DataHandler(cfg)
        pretrain_df = handler.get_pretraining_train_dataframe()

        assert pretrain_df is not None
        assert len(pretrain_df) == 3
        assert pretrain_df.iloc[0]["text"] == "cod: description-0"
        assert pretrain_df.iloc[0]["label"] == "A00.000"

    def test_get_pretraining_train_dataframe_applies_pretrain_dataset_size(
        self, tmp_path: Path
    ) -> None:
        """Pretraining loader should honor pretrain_dataset_size sampling."""
        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist(masterlist_path, num_rows=10)

        cfg = Config(
            pretrain_enabled=True,
            pretrain_masterlist_path=str(masterlist_path),
            pretrain_masterlist_sheet_name="Masterlist",
            pretrain_dataset_size=0.5,
            data_sources=[],
        )
        handler = DataHandler(cfg)
        pretrain_df = handler.get_pretraining_train_dataframe()

        assert pretrain_df is not None
        assert len(pretrain_df) == 5

    def test_select_upsample_targets_returns_quantile_minority_targets(self) -> None:
        """Minority selector should return per-label target counts."""
        targets = select_upsample_targets(
            _balance_df(),
            label_column="label",
            target_quantile=1.0,
            inverse_power=1.0,
            budget_ratio=1.0,
        )
        assert targets == {"B00": 2, "C00": 2}

    def test_upsample_adds_rows_for_targeted_class(self) -> None:
        """Upsample should append synthetic rows until target count is reached."""
        balanced = upsample(
            df=_balance_df(),
            label_column="label",
            target_counts={"B00": 2},
            seed=3,
        )
        counts = balanced["label"].value_counts().to_dict()
        assert counts["B00"] == 2
        assert len(balanced) == 5
        assert balanced.iloc[-1]["text"].endswith(" | age: 3 | sex: female")

    def test_upsample_samples_class_rows_before_repeating(self) -> None:
        """Upsample should draw from all available class rows before repeating."""
        source = pd.DataFrame(
            {
                "text": [
                    "cod: major-1 | age: 1 | sex: male",
                    "cod: major-2 | age: 2 | sex: male",
                    "cod: minor-a | age: 3 | sex: female",
                    "cod: minor-b | age: 4 | sex: female",
                ],
                "label": ["A00", "A00", "B00", "B00"],
            }
        )
        balanced = upsample(
            df=source,
            label_column="label",
            target_counts={"B00": 4},
            seed=3,
        )
        synthetic_rows = balanced.iloc[len(source) :]
        assert len(synthetic_rows) == 2
        assert set(synthetic_rows["text"].tolist()) == {
            "cod: minor-a | age: 3 | sex: female",
            "cod: minor-b | age: 4 | sex: female",
        }

    def test_select_upsample_targets_supports_dynamic_inverse_scaling(self) -> None:
        """Dynamic upsampling should preserve minority class ordering and cap at q."""
        df = pd.DataFrame(
            {
                "text": ["cod: t | age: 1 | sex: male"] * 1101,
                "label": ["major"] * 1000 + ["mid"] * 100 + ["rare"],
            }
        )
        targets = select_upsample_targets(
            df,
            label_column="label",
            target_quantile=1.0,
            inverse_power=1.0,
            budget_ratio=1.0,
        )
        assert targets["mid"] > targets["rare"]
        assert targets["mid"] <= 1000
        assert targets["rare"] <= 1000

    def test_manipulate_classes_perturbs_targets_without_changing_row_count(
        self,
    ) -> None:
        """Manipulation-only mode should mutate selected labels in place."""
        cfg = Config(text_field_separator=" | ")
        source = _balance_df()
        manipulated = manipulate_classes(
            cfg=cfg,
            df=source,
            text_column="text",
            label_column="label",
            target_labels=["A00"],
            perturbation_names=["delete_random_char"],
            perturbations_per_sample=1,
            sample_fraction=1.0,
            seed=9,
        )
        assert len(manipulated) == len(source)
        assert (
            manipulated[manipulated["label"] == "A00"]["text"].tolist()
            != source[source["label"] == "A00"]["text"].tolist()
        )
        assert (
            manipulated[manipulated["label"] == "B00"]["text"].tolist()
            == source[source["label"] == "B00"]["text"].tolist()
        )
        assert (
            manipulated[manipulated["label"] == "C00"]["text"].tolist()
            == source[source["label"] == "C00"]["text"].tolist()
        )

    def test_split_dataframe_uses_configured_sizes(self) -> None:
        """Split sizes should be respected for train/validation/test output."""
        cfg = Config(train_size=0.6, val_size=0.2, test_size=0.2)
        handler = DataHandler(cfg)
        df = _processed_df(num_rows=20)
        splits = handler.split_dataframe(df)
        assert len(splits.train) == 12
        assert len(splits.val) == 4
        assert len(splits.test) == 4

    def test_get_splits_processes_data_when_missing(self, tmp_path: Path) -> None:
        """Handler should build processed data when output file is missing."""
        raw_df = pd.DataFrame(
            [
                [f"text-{idx}", f"A{idx:03d}", "", "1", str(idx), f"RID-{idx:03d}"]
                for idx in range(20)
            ]
        )
        raw_path = tmp_path / "raw.csv"
        raw_df.to_csv(raw_path, index=False)

        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(tmp_path / "processed"),
            processed_filename="training.csv",
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="raw.csv",
                    mapping_id="test_mapping",
                )
            ],
            train_size=0.5,
            val_size=0.25,
            test_size=0.25,
            dataset_size=1.0,
        )
        handler = DataHandler(cfg, mapping_registry={"test_mapping": _make_mapping()})
        splits = handler.get_splits()
        assert handler.processed_path.exists()
        assert handler.processed_metadata_path.exists()
        assert len(splits.train) == 10
        assert len(splits.val) == 5
        assert len(splits.test) == 5

    def test_get_splits_uses_existing_processed_file(self, tmp_path: Path) -> None:
        """Handler should load and split existing processed data."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        _processed_df(num_rows=20).to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.7,
            val_size=0.2,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())
        splits = handler.get_splits()
        assert len(splits.train) == 14
        assert len(splits.val) == 4
        assert len(splits.test) == 2

    def test_get_splits_ignores_balance_only_metadata_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Balance-only config changes should not invalidate processed cache."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        _processed_df(num_rows=20).to_csv(processed_path, index=False)

        cfg_base = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.7,
            val_size=0.2,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
            balance_strategy="upsample",
            balance_target_quantile=0.6,
            balance_upsample_budget_ratio=0.25,
            balance_base_perturbation_rate=0.2,
        )
        handler_base = DataHandler(cfg_base)
        legacy_like_metadata = handler_base._build_processing_metadata()
        legacy_like_metadata["balance_strategy"] = "upsample"
        legacy_like_metadata["balance_target_quantile"] = 0.6
        legacy_like_metadata["balance_upsample_budget_ratio"] = 0.25
        legacy_like_metadata["balance_base_perturbation_rate"] = 0.2
        handler_base._write_processing_metadata(legacy_like_metadata)

        monkeypatch.setattr(
            data_handler_module,
            "build_processed_dataset",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Processed data should not be rebuilt.")
            ),
        )

        cfg_changed = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.7,
            val_size=0.2,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
            balance_strategy="none",
            balance_target_quantile=0.9,
            balance_upsample_budget_ratio=0.0,
            balance_base_perturbation_rate=0.0,
        )
        handler_changed = DataHandler(cfg_changed)
        splits = handler_changed.get_splits()

        assert len(splits.train) == 14
        assert len(splits.val) == 4
        assert len(splits.test) == 2

    def test_get_splits_reprocesses_when_metadata_missing(self, tmp_path: Path) -> None:
        """Missing metadata should trigger a processed rebuild."""
        raw_df = pd.DataFrame(
            [["cholera", "A00.000", "J18.100", "1", "20", "RID-001", "R99.900"]]
        )
        raw_path = tmp_path / "raw.csv"
        raw_df.to_csv(raw_path, index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        pd.DataFrame(
            {
                "source_id": ["stale"],
                "record_id": ["old"],
                "source_path": ["old.csv"],
                "text": ["cod: stale | age: 99 | sex: male"],
                "y_codes": [["A99.999"]],
                "label": ["A99.999"],
            }
        ).to_csv(processed_path, index=False)

        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="raw.csv",
                    mapping_id="test_mapping",
                )
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=2,
            label_separator=",",
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
        )
        handler = DataHandler(
            cfg,
            mapping_registry={"test_mapping": _make_mapping(multi_code_cols=[2, 6])},
        )
        _ = handler.get_splits()

        refreshed = pd.read_csv(processed_path)
        assert refreshed.iloc[0]["source_id"] == "csv_source"
        assert refreshed.iloc[0]["label"] == "J18.100,R99.900"
        assert handler.processed_metadata_path.exists()

    def test_get_splits_reprocesses_when_source_file_changes(
        self, tmp_path: Path
    ) -> None:
        """Source-file signature changes should invalidate processed cache."""
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            [["cholera", "A00.000", "J18.100", "1", "20", "RID-001", "R99.900"]]
        ).to_csv(raw_path, index=False)

        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(tmp_path / "processed"),
            processed_filename="training.csv",
            data_sources=[
                DataSourceConfig(
                    source_id="csv_source",
                    path="raw.csv",
                    mapping_id="test_mapping",
                )
            ],
            training_input=["cod", "age", "sex"],
            max_label_count=1,
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
        )
        handler = DataHandler(
            cfg,
            mapping_registry={"test_mapping": _make_mapping(multi_code_cols=[2])},
        )
        _ = handler.get_splits()
        first_processed = pd.read_csv(handler.processed_path)
        assert first_processed.iloc[0]["label"] == "J18.100"

        pd.DataFrame(
            [["cholera", "A00.000", "B01.001", "1", "20", "RID-001", ""]]
        ).to_csv(raw_path, index=False)
        _ = handler.get_splits()
        second_processed = pd.read_csv(handler.processed_path)
        assert second_processed.iloc[0]["label"] == "B01.001"

    def test_get_splits_applies_dataset_size_sampling(self, tmp_path: Path) -> None:
        """dataset_size should downsample data before split."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        _processed_df(num_rows=20).to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            dataset_size=0.5,
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())
        splits = handler.get_splits()
        assert len(splits.train) == 8
        assert len(splits.val) == 1
        assert len(splits.test) == 1

    def test_get_splits_applies_base_perturbation_to_all_labels(
        self, tmp_path: Path
    ) -> None:
        """Base perturbation should affect all labels after balancing."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        source_df = pd.DataFrame(
            {
                "source_id": ["src"] * 3,
                "record_id": ["RID-001", "RID-002", "RID-003"],
                "source_path": ["sample.csv"] * 3,
                "text": [
                    "cod: alpha | age: 1 | sex: male",
                    "cod: beta | age: 2 | sex: male",
                    "cod: gamma | age: 3 | sex: female",
                ],
                "y_codes": [["A00"], ["A00"], ["B00"]],
                "label": ["A00", "A00", "B00"],
            }
        )
        source_df.to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            dataset_size=1.0,
            data_sources=[],
            balance_strategy="none",
            balance_target_quantile=1.0,
            balance_perturbations=["delete_random_char"],
            balance_base_perturbation_rate=1.0,
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())
        splits = handler.get_splits()

        source_text_by_id = source_df.set_index("record_id")["text"].to_dict()
        train_text_by_id = splits.train.set_index("record_id")["text"].to_dict()
        assert len(splits.train) == len(source_df)
        assert train_text_by_id["RID-001"] != source_text_by_id["RID-001"]
        assert train_text_by_id["RID-002"] != source_text_by_id["RID-002"]
        assert train_text_by_id["RID-003"] != source_text_by_id["RID-003"]

    def test_get_splits_rejects_processed_data_with_empty_labels(
        self, tmp_path: Path
    ) -> None:
        """Processed data must contain at least one non-empty label."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        pd.DataFrame(
            {
                "source_id": ["src1", "src2"],
                "record_id": ["RID-001", "RID-002"],
                "source_path": ["sample.csv", "sample.csv"],
                "text": [
                    "cod: one | age: 1 | sex: male",
                    "cod: two | age: 2 | sex: female",
                ],
                "y_codes": [[], []],
                "label": ["", ""],
            }
        ).to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())

        with pytest.raises(ValueError, match="no non-empty values"):
            handler.get_splits()

    def test_split_dataframe_uses_data_seed_when_configured(self) -> None:
        """Configured data_seed should control deterministic split ordering."""
        cfg = Config(train_size=0.6, val_size=0.2, test_size=0.2, seed=42, data_seed=7)
        handler = DataHandler(cfg)
        df = _processed_df(num_rows=20)
        splits = handler.split_dataframe(df)

        expected_train, expected_holdout = train_test_split(
            df,
            test_size=0.4,
            random_state=7,
            shuffle=True,
        )
        expected_val, expected_test = train_test_split(
            expected_holdout,
            test_size=0.5,
            random_state=7,
            shuffle=True,
        )
        assert (
            splits.train["record_id"].tolist()
            == expected_train.reset_index(drop=True)["record_id"].tolist()
        )
        assert (
            splits.val["record_id"].tolist()
            == expected_val.reset_index(drop=True)["record_id"].tolist()
        )
        assert (
            splits.test["record_id"].tolist()
            == expected_test.reset_index(drop=True)["record_id"].tolist()
        )

    def test_split_dataframe_rejects_invalid_split_sum(self) -> None:
        """Invalid split totals should fail with a clear error."""
        cfg = Config(train_size=0.7, val_size=0.2, test_size=0.2)
        handler = DataHandler(cfg)
        with pytest.raises(ValueError):
            handler.split_dataframe(_processed_df(num_rows=20))

    def test_ensure_processed_serializes_parallel_rebuilds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent ensure_processed calls should rebuild shared cache once."""
        cfg = Config(
            data_processed_dir=str(tmp_path / "processed"),
            processed_filename="training.csv",
            data_sources=[],
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
        )
        handler = DataHandler(cfg)
        rebuilt_frame = _processed_df(num_rows=4)
        build_calls = {"count": 0}

        def fake_build_processed_dataset(
            *args: object, **kwargs: object
        ) -> pd.DataFrame:
            build_calls["count"] += 1
            time.sleep(0.2)
            return rebuilt_frame.copy()

        monkeypatch.setattr(
            data_handler_module,
            "build_processed_dataset",
            fake_build_processed_dataset,
        )

        errors: list[Exception] = []
        results: list[pd.DataFrame] = []

        def run_worker() -> None:
            try:
                results.append(handler.ensure_processed(force_reprocess=False))
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=run_worker)
        second = threading.Thread(target=run_worker)
        first.start()
        second.start()
        first.join()
        second.join()

        assert not errors
        assert build_calls["count"] == 1
        assert len(results) == 2
        assert handler.processed_path.exists()
        assert handler.processed_metadata_path.exists()
