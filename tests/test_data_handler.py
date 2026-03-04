from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from codllm.config import Config, DataSourceConfig
from codllm.data_handler import (
    DataHandler,
    DatasetMapping,
    _build_text,
    _build_y,
    build_and_save_processed_dataset,
    build_processed_dataset,
    load_dataset,
    load_source_dataset,
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


class TestBuildY:
    def test_build_y_limits_output_to_one_label_by_default(self) -> None:
        """y should be capped to one code for current training objective."""
        mapping = _make_mapping(multi_code_cols=[2, 6])
        row = _row("text", "A00", "J18", "1", "20", "RID-001", "R99")
        assert _build_y(row, mapping) == ["J18"]

    def test_build_y_can_return_multiple_labels_for_future_use(self) -> None:
        """y should support multiple labels when max_labels is increased."""
        mapping = _make_mapping(multi_code_cols=[2, 6])
        row = _row("text", "A00", "J18", "1", "20", "RID-001", "R99")
        assert _build_y(row, mapping, max_labels=2) == ["J18", "R99"]


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
            mapping_registry={"test_mapping": _make_mapping(multi_code_cols=[2, 6])},
        )
        _ = handler.get_splits()
        first_processed = pd.read_csv(handler.processed_path)
        assert first_processed.iloc[0]["label"] == "J18.100"

        pd.DataFrame(
            [["cholera", "A00.000", "B01.001", "1", "20", "RID-001", "R99.900"]]
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
