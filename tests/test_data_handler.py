from pathlib import Path
import threading
import time

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import codllm.data.handler as data_handler_module
from codllm.config import Config, DataSourceConfig
from codllm.input import COPENHAGEN_MAPPING, build_synthetic_multicod_rows
from codllm.data.handler import (
    DataHandler,
    DataSplits,
    DatasetMapping,
    MAPPING_REGISTRY,
    _build_text,
    _build_y,
    build_and_save_processed_dataset,
    build_processed_dataset,
    load_dataset,
    load_source_dataset,
    manipulate_classes,
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


def _write_masterlist_with_transfer(
    path: Path,
    master_codes: list[str],
    transfer_pairs: list[tuple[str, str]],
) -> None:
    """Write a compact masterlist workbook with transfer mapping sheet."""
    master_rows = []
    for idx, code in enumerate(master_codes, start=1):
        master_rows.append(
            {
                "IDMasterlist": idx,
                "ICD10h": code,
                "ICD10": f"{code[:4]}{code[4]}" if len(code) >= 5 else code,
                "icd10_2levelCATEGORY": "Category",
                "ICD10_2levelCAUSE": f"Cause {idx}",
                "ICD10h_DESCRIPTION": f"description-{idx}",
                "HistCat": "Hist",
                "DoNotUse": 0,
                "NotForUnderlying": 0,
                "GenderSpecific": 0,
            }
        )

    transfer_rows = []
    for idx, (old_code, new_code) in enumerate(transfer_pairs, start=1):
        transfer_rows.append(
            {
                "ID2024Transfer": idx,
                "IDoct2020Masterlist": idx,
                "ICD10h_oct2020": old_code,
                "ICD10h2024": new_code,
            }
        )

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(master_rows).to_excel(writer, sheet_name="Masterlist", index=False)
        pd.DataFrame(transfer_rows).to_excel(
            writer, sheet_name="2020to2024transfer", index=False
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

    def test_build_text_uses_configured_input_prefixes(self) -> None:
        """Text should use configured field prefixes for each input field."""
        mapping = _make_mapping()
        row = _row("cholera", "A00", "", "1", "2.4", "RID-001")
        cfg = Config(
            input_field_prefixes={
                "cod": "cause=",
                "age": "years=",
                "sex": "gender=",
            }
        )

        result = _build_text(
            row,
            mapping,
            ["cod", "age", "sex"],
            field_separator=cfg.text_field_separator,
            input_field_prefixes=cfg.input_field_prefixes,
        )

        assert result == "cause=cholera | years=2.4 | gender=male"

    def test_build_text_omits_cod_prefix_for_cod_only_input(self) -> None:
        """COD-only text should contain only the normalized COD value."""
        mapping = _make_mapping()
        row = _row("cholera", "A00", "", "1", "2.4", "RID-001")

        result = _build_text(row, mapping, ["cod"])

        assert result == "cholera"


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
        assert result.iloc[1]["y"] == ["A01"]

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

    def test_load_source_dataset_omits_cod_prefix_for_cod_only_input(
        self, tmp_path: Path
    ) -> None:
        """Source loader should emit bare COD text when COD is the only input field."""
        csv_path = tmp_path / "sample.csv"
        _sample_df().to_csv(csv_path, index=False)
        source = DataSourceConfig(
            source_id="test_source", path=str(csv_path), mapping_id="test_mapping"
        )
        mapping = _make_mapping()

        result = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=["cod"],
            max_labels=1,
            data_raw_dir="",
        )

        assert result.iloc[0]["text"] == "cholera"

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

    def test_load_source_dataset_uses_single_code_when_max_labels_is_one(
        self, tmp_path: Path
    ) -> None:
        """Single-label processing should use the configured single cause column."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                ["text-one", "A00.000", "J18.100", "1", "20", "RID-001", "R99.900"],
                ["text-two", "B01.001", "B02.002", "2", "21", "RID-002", ""],
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

        assert result["record_id"].tolist() == ["RID-001", "RID-002"]
        assert result["y_codes"].tolist() == [["A00.000"], ["B01.001"]]
        assert result["label"].tolist() == ["A00.000", "B01.001"]

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
            label_harmonization_enabled=False,
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
            label_harmonization_enabled=False,
        )
        mapping = _make_mapping()
        result = build_processed_dataset(
            cfg, mapping_registry={"test_mapping": mapping}
        )
        assert len(result) == 4
        assert sorted(result["source_id"].unique()) == ["csv_source", "xlsx_source"]
        assert set(result["label"]) == {"A00", "A01"}

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
            label_harmonization_enabled=False,
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
            label_harmonization_enabled=False,
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
            input_field_prefixes={
                "cod": "cause=",
                "age": "years=",
                "sex": "gender=",
            },
            max_label_count=2,
            label_separator=",",
            label_harmonization_enabled=False,
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

    def test_get_splits_shuffles_multicod_label_order(self, tmp_path: Path) -> None:
        """Multi-COD labels should be shuffled deterministically from data_seed."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                [
                    "text",
                    "A00.000",
                    "J18.100",
                    "1",
                    "20",
                    "RID-001",
                    "R99.900",
                    "I10.000",
                ],
                ["other", "K11.111", "", "1", "21", "RID-002", "", ""],
            ]
        ).to_csv(csv_path, index=False)
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
            max_label_count=3,
            label_separator=",",
            data_seed=7,
            multicod_shuffle_labels=True,
            data_processed_dir=str(tmp_path / "processed"),
            processed_filename="training.csv",
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            base_perturbation_rate=0.0,
            label_harmonization_enabled=False,
        )
        mapping = _make_mapping(multi_code_cols=[2, 6, 7])

        handler = DataHandler(cfg, mapping_registry={"test_mapping": mapping})
        splits = handler.get_splits()
        row = splits.train[splits.train["record_id"] == "RID-001"].iloc[0]

        assert row["y_codes"] == ["I10.000", "J18.100", "R99.900"]
        assert row["label"] == "I10.000,J18.100,R99.900"

    def test_get_splits_merges_single_cods_within_source_for_train_only(
        self, tmp_path: Path
    ) -> None:
        """Synthetic multi-COD rows should merge single-label rows without crossing sources."""
        source_one_path = tmp_path / "source_one.csv"
        source_two_path = tmp_path / "source_two.csv"
        pd.DataFrame(
            [
                ["alpha", "A00.000", "", "1", "20", "ONE-001"],
                ["beta", "B01.001", "", "1", "21", "ONE-002"],
            ]
        ).to_csv(source_one_path, index=False)
        pd.DataFrame(
            [
                ["gamma", "C02.002", "", "2", "30", "TWO-001"],
                ["delta", "D03.003", "", "2", "31", "TWO-002"],
            ]
        ).to_csv(source_two_path, index=False)
        cfg = Config(
            data_raw_dir=str(tmp_path),
            data_sources=[
                DataSourceConfig(
                    source_id="source_one",
                    path="source_one.csv",
                    mapping_id="test_mapping",
                ),
                DataSourceConfig(
                    source_id="source_two",
                    path="source_two.csv",
                    mapping_id="test_mapping",
                ),
            ],
            training_input=["cod", "age", "sex"],
            input_field_prefixes={
                "cod": "cause=",
                "age": "years=",
                "sex": "gender=",
            },
            max_label_count=2,
            label_separator=",",
            multicod_shuffle_labels=False,
            multicod_synthetic_ratio=1.0,
            multicod_synthetic_source_scope="within_source",
            multicod_synthetic_text_separators=["; "],
            data_processed_dir=str(tmp_path / "processed"),
            processed_filename="training.csv",
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            base_perturbation_rate=0.0,
            label_harmonization_enabled=False,
        )
        mapping = _make_mapping(multi_code_cols=[])

        handler = DataHandler(cfg, mapping_registry={"test_mapping": mapping})
        splits = handler.get_splits()

        result = splits.train
        synthetic = result[result["source_id"].str.startswith("synthetic_multicod:")]
        persisted = pd.read_csv(handler.processed_path)
        assert len(result) == 8
        assert len(synthetic) == 4
        assert len(persisted) == 4
        assert not persisted["source_id"].str.startswith("synthetic_multicod:").any()
        assert splits.val.empty
        assert splits.test.empty
        assert {frozenset(codes) for codes in synthetic["y_codes"].tolist()} <= {
            frozenset({"A00.000", "B01.001"}),
            frozenset({"C02.002", "D03.003"}),
        }
        for _, row in synthetic.iterrows():
            text = row["text"]
            codes = set(row["y_codes"])
            assert text.startswith("cause=")
            assert "years=" in text
            assert "gender=" in text
            if codes == {"A00.000", "B01.001"}:
                assert "alpha" in text and "beta" in text
            if codes == {"C02.002", "D03.003"}:
                assert "gamma" in text and "delta" in text

    def test_synthetic_multicod_rows_follow_existing_cardinality_distribution(
        self,
    ) -> None:
        """Synthetic multi-COD rows should mirror observed multi-COD cardinality."""
        cfg = Config(
            max_label_count=3,
            label_separator=",",
            multicod_synthetic_ratio=5.0,
            multicod_synthetic_source_scope="within_source",
            multicod_synthetic_text_separators=[", "],
            data_seed=17,
        )
        source = pd.DataFrame(
            {
                "source_id": ["real"] * 6,
                "record_id": [f"RID-{idx}" for idx in range(6)],
                "source_path": ["sample.csv"] * 6,
                "text": [
                    "cod: alpha | age: 1 | sex: male",
                    "cod: beta | age: 1 | sex: male",
                    "cod: gamma | age: 1 | sex: male",
                    "cod: delta | age: 1 | sex: male",
                    "cod: alpha, beta | age: 1 | sex: male",
                    "cod: gamma, delta | age: 1 | sex: male",
                ],
                "y_codes": [
                    ["A00"],
                    ["B00"],
                    ["C00"],
                    ["D00"],
                    ["A00", "B00"],
                    ["C00", "D00"],
                ],
                "label": ["A00", "B00", "C00", "D00", "A00,B00", "C00,D00"],
            }
        )

        synthetic = build_synthetic_multicod_rows(source, cfg)

        assert len(synthetic) == 20
        assert synthetic["y_codes"].map(len).eq(2).all()

    def test_synthetic_multicod_rows_vary_text_separators(self) -> None:
        """Synthetic multi-COD text should sample from configured separator variants."""
        cfg = Config(
            max_label_count=2,
            label_separator=",",
            multicod_synthetic_ratio=4.0,
            multicod_synthetic_text_separators=[" ~~ ", " && "],
            data_seed=23,
        )
        source = pd.DataFrame(
            {
                "source_id": ["real"] * 4,
                "record_id": [f"RID-{idx}" for idx in range(4)],
                "source_path": ["sample.csv"] * 4,
                "text": [
                    "cod: alpha | age: 1 | sex: male",
                    "cod: beta | age: 1 | sex: male",
                    "cod: gamma | age: 1 | sex: male",
                    "cod: delta | age: 1 | sex: male",
                ],
                "y_codes": [["A00"], ["B00"], ["C00"], ["D00"]],
                "label": ["A00", "B00", "C00", "D00"],
            }
        )

        synthetic = build_synthetic_multicod_rows(source, cfg)
        synthetic_texts = synthetic["text"].tolist()

        assert len(synthetic) == 16
        assert any(" ~~ " in text for text in synthetic_texts)
        assert any(" && " in text for text in synthetic_texts)

    def test_load_source_dataset_excludes_rows_above_max_label_count(
        self, tmp_path: Path
    ) -> None:
        """Rows with more than max_labels should be removed from processed output."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                [
                    "text-one",
                    "A00.000",
                    "J18.100",
                    "1",
                    "20",
                    "RID-001",
                    "R99.900",
                    "I10.000",
                ],
                ["text-two", "B01.001", "B01.001", "1", "21", "RID-002", "", ""],
            ]
        ).to_csv(csv_path, index=False)

        source = DataSourceConfig(
            source_id="test_source", path=str(csv_path), mapping_id="test_mapping"
        )
        mapping = _make_mapping(multi_code_cols=[2, 6, 7])
        result = load_source_dataset(
            source=source,
            mapping=mapping,
            training_input=["cod", "age", "sex"],
            max_labels=2,
            data_raw_dir="",
        )
        assert len(result) == 1
        assert result.iloc[0]["record_id"] == "RID-002"
        assert result.iloc[0]["label"] == "B01.001"

    def test_build_processed_dataset_harmonizes_labels_against_masterlist(
        self, tmp_path: Path
    ) -> None:
        """Harmonization should map transfer labels, pad suffixes, and drop unknowns."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame(
            [
                ["text-a", "A09.001", "", "1", "20", "RID-001"],
                ["text-b", "Q36.9", "", "1", "21", "RID-002"],
                ["text-c", "Q36.9001", "", "1", "22", "RID-003"],
                ["text-d", "X99.999", "", "1", "23", "RID-004"],
            ]
        ).to_csv(csv_path, index=False)

        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist_with_transfer(
            masterlist_path,
            master_codes=["A09.052", "Q36.900"],
            transfer_pairs=[("A09.001", "A09.052")],
        )

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
            max_label_count=1,
            label_harmonization_masterlist_path=str(masterlist_path),
            label_harmonization_masterlist_sheet_name="Masterlist",
            label_harmonization_transfer_sheet_name="2020to2024transfer",
        )
        mapping = _make_mapping(multi_code_cols=[])

        result = build_processed_dataset(
            cfg=cfg,
            mapping_registry={"test_mapping": mapping},
        )

        assert result["record_id"].tolist() == ["RID-001", "RID-002", "RID-003"]
        assert result["label"].tolist() == ["A09.052", "Q36.900", "Q36.900"]
        assert result["y_codes"].tolist() == [
            ["A09.052"],
            ["Q36.900"],
            ["Q36.900"],
        ]

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
            label_harmonization_enabled=False,
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
            training_input=["cod"],
            data_sources=[],
            pretrain_upsample_enabled=False,
        )
        handler = DataHandler(cfg)
        pretrain_df = handler.get_pretraining_train_dataframe()
        upsampling_metrics = handler.get_pretraining_upsampling_metrics()

        assert pretrain_df is not None
        assert len(pretrain_df) == 3
        assert pretrain_df.iloc[0]["text"] == "description-0"
        assert pretrain_df.iloc[0]["label"] == "A00.000"
        assert upsampling_metrics is not None
        assert upsampling_metrics["enabled"] is False
        assert upsampling_metrics["rows_before"] == 3
        assert upsampling_metrics["rows_after"] == 3
        assert upsampling_metrics["rows_added"] == 0

    def test_get_pretraining_train_dataframe_upsamples_masterlist_and_tracks_metrics(
        self, tmp_path: Path
    ) -> None:
        """Pretraining rows should upsample to target per label and report metrics."""
        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist(masterlist_path, num_rows=2)

        cfg = Config(
            pretrain_enabled=True,
            pretrain_masterlist_path=str(masterlist_path),
            pretrain_masterlist_sheet_name="Masterlist",
            pretrain_upsample_enabled=True,
            pretrain_upsample_target_per_label=3,
            pretrain_upsample_perturbations=["delete_random_char"],
            pretrain_upsample_perturbations_per_sample=1,
            training_input=["cod"],
            data_sources=[],
        )
        handler = DataHandler(cfg)
        pretrain_df = handler.get_pretraining_train_dataframe()
        upsampling_metrics = handler.get_pretraining_upsampling_metrics()

        assert pretrain_df is not None
        assert len(pretrain_df) == 6
        assert pretrain_df["label"].value_counts().to_dict() == {
            "A00.000": 3,
            "A01.000": 3,
        }
        assert upsampling_metrics is not None
        assert upsampling_metrics["enabled"] is True
        assert upsampling_metrics["target_examples_per_label"] == 3
        assert upsampling_metrics["rows_before"] == 2
        assert upsampling_metrics["rows_after"] == 6
        assert upsampling_metrics["rows_added"] == 4
        assert upsampling_metrics["synthetic_rows"] == 4
        assert upsampling_metrics["labels_upsampled"] == 2
        assert upsampling_metrics["labels_below_target_before"] == 2
        assert upsampling_metrics["labels_below_target_after"] == 0
        assert upsampling_metrics["perturbations_per_sample"] == 1
        assert upsampling_metrics["perturbation_applications"] == 4
        assert upsampling_metrics["perturbed_rows"] == 4
        assert upsampling_metrics["perturbation_rate"] == pytest.approx(1.0)
        assert upsampling_metrics["label_count_summary_before"]["min"] == pytest.approx(
            1.0
        )
        assert upsampling_metrics["label_count_summary_after"]["min"] == pytest.approx(
            3.0
        )

    def test_get_pretraining_train_dataframe_adds_multicod_synthetic_rows(
        self, tmp_path: Path
    ) -> None:
        """Pretraining can synthesize multi-COD rows by merging masterlist causes."""
        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist(masterlist_path, num_rows=4)

        cfg = Config(
            pretrain_enabled=True,
            pretrain_masterlist_path=str(masterlist_path),
            pretrain_masterlist_sheet_name="Masterlist",
            pretrain_upsample_enabled=False,
            pretrain_multicod_synthetic_ratio=0.5,
            pretrain_multicod_synthetic_text_separators=[" + "],
            max_label_count=3,
            training_input=["cod"],
            data_sources=[],
        )
        handler = DataHandler(cfg)
        pretrain_df = handler.get_pretraining_train_dataframe()
        multicod_metrics = handler.get_pretraining_multicod_metrics()

        assert pretrain_df is not None
        synthetic = pretrain_df[
            pretrain_df["source_id"].str.startswith("synthetic_pretrain_multicod:")
        ]
        assert len(pretrain_df) == 6
        assert len(synthetic) == 2
        assert synthetic["text"].str.contains(r".+ \+ .+").all()
        assert synthetic["y_codes"].map(len).between(2, 3).all()
        assert synthetic["label"].str.contains(",").all()
        assert multicod_metrics is not None
        assert multicod_metrics["enabled"] is True
        assert multicod_metrics["ratio"] == pytest.approx(0.5)
        assert multicod_metrics["rows_before"] == 4
        assert multicod_metrics["rows_after"] == 6
        assert multicod_metrics["synthetic_rows"] == 2
        assert multicod_metrics["label_count_distribution"]["1"] == 4
        assert (
            sum(
                count
                for label_count, count in multicod_metrics[
                    "label_count_distribution"
                ].items()
                if label_count != "1"
            )
            == 2
        )

    def test_get_masterlist_label_vocabulary_loads_sorted_unique_labels(
        self, tmp_path: Path
    ) -> None:
        """Masterlist label vocabulary should expose sorted unique ICD10h labels."""
        masterlist_path = tmp_path / "ICD10h_Masterlist_2024.xlsx"
        _write_masterlist(masterlist_path, num_rows=4)

        cfg = Config(
            label_harmonization_masterlist_path=str(masterlist_path),
            label_harmonization_masterlist_sheet_name="Masterlist",
            training_input=["cod"],
            data_sources=[],
        )
        handler = DataHandler(cfg)
        labels = handler.get_masterlist_label_vocabulary()

        assert labels == ["A00.000", "A01.000", "A02.000", "A03.000"]

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
            perturbation_mean=0.1,
            perturbation_variance=0.0,
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

    def test_manipulate_classes_uses_configured_cod_prefix(self) -> None:
        """Manipulation should find the configured COD segment prefix."""
        cfg = Config(
            text_field_separator=" | ",
            training_input=["cod", "age", "sex"],
            input_field_prefixes={
                "cod": "cause=",
                "age": "years=",
                "sex": "gender=",
            },
        )
        source = pd.DataFrame(
            {
                "text": ["cause=alpha | years=1 | gender=male"],
                "label": ["A00"],
            }
        )

        manipulated = manipulate_classes(
            cfg=cfg,
            df=source,
            text_column="text",
            label_column="label",
            target_labels=["A00"],
            perturbation_names=["delete_random_char"],
            perturbation_mean=0.2,
            perturbation_variance=0.0,
            sample_fraction=1.0,
            seed=9,
        )

        assert manipulated.iloc[0]["text"] != source.iloc[0]["text"]
        assert manipulated.iloc[0]["text"].startswith("cause=")
        assert "years=1 | gender=male" in manipulated.iloc[0]["text"]

    def test_apply_balance_policy_tracks_visualization_metrics(self) -> None:
        """Balance policy should expose summary metrics for W&B visualizations."""
        cfg = Config(
            text_field_separator=" | ",
            balance_strategy="floor",
            balance_floor=4,
            balance_floor_decay=0.0,
            balance_perturbations=["delete_random_char"],
            balance_perturbation_mean=0.1,
            balance_perturbation_variance=0.0,
            base_perturbation_rate=1.0,
            base_perturbations=["delete_random_char"],
            base_perturbation_mean=0.1,
            base_perturbation_variance=0.0,
        )
        handler = DataHandler(cfg)
        source = _balance_df()
        balanced = handler._apply_balance_policy(source)
        metrics = handler.get_training_balance_metrics()

        assert metrics is not None
        assert len(balanced) > len(source)
        assert metrics["enabled"] is True
        assert metrics["strategy"] == "floor"
        assert metrics["rows_before"] == len(source)
        assert metrics["rows_after"] == len(balanced)
        assert metrics["rows_added"] == len(balanced) - len(source)
        assert metrics["base_perturbed_rows"] > 0
        assert metrics["label_distribution_before"]["A00"] == 2
        assert metrics["label_distribution_after"]["B00"] >= 1

    def test_apply_balance_policy_does_not_floor_upsample_multicod_rows(self) -> None:
        """Floor balancing should upsample atomic labels without expanding multi-COD rows."""
        cfg = Config(
            text_field_separator=" | ",
            label_separator=",",
            balance_strategy="floor",
            balance_floor=3,
            balance_floor_decay=0.0,
            balance_perturbations=[],
            base_perturbation_rate=0.0,
            max_label_count=2,
        )
        handler = DataHandler(cfg)
        source = pd.DataFrame(
            {
                "source_id": [
                    "real",
                    "real",
                    "real",
                    "synthetic_multicod:real",
                ],
                "text": [
                    "cod: alpha",
                    "cod: beta",
                    "cod: alpha, beta",
                    "cod: alpha, beta",
                ],
                "y_codes": [
                    ["A00"],
                    ["B00"],
                    ["A00", "B00"],
                    ["A00", "B00"],
                ],
                "label": ["A00", "B00", "A00,B00", "A00,B00"],
            }
        )

        balanced = handler._apply_balance_policy(source)
        metrics = handler.get_training_balance_metrics()

        assert metrics is not None
        assert len(balanced) == 8
        assert balanced["label"].value_counts().to_dict() == {
            "A00": 3,
            "B00": 3,
            "A00,B00": 2,
        }
        assert metrics["rows_added"] == 4
        assert metrics["upsample_eligible_rows"] == 2
        assert metrics["upsample_excluded_multicod_rows"] == 2
        assert metrics["upsample_excluded_synthetic_multicod_rows"] == 1

    def test_manipulate_classes_scales_perturbation_count_by_cod_length(
        self,
    ) -> None:
        """Length-scaled perturbation settings should affect longer COD text more."""
        cfg = Config(text_field_separator=" | ", training_input=["cod", "age", "sex"])
        source = pd.DataFrame(
            {
                "text": [
                    "cod: ab | age: 1 | sex: male",
                    "cod: abcdefghij | age: 1 | sex: male",
                ],
                "label": ["A00", "A00"],
            }
        )

        manipulated = manipulate_classes(
            cfg=cfg,
            df=source,
            text_column="text",
            label_column="label",
            target_labels=["A00"],
            perturbation_names=["delete_random_char"],
            perturbation_mean=0.5,
            perturbation_variance=0.0,
            sample_fraction=1.0,
            seed=9,
        )

        short_delta = len(source.iloc[0]["text"]) - len(manipulated.iloc[0]["text"])
        long_delta = len(source.iloc[1]["text"]) - len(manipulated.iloc[1]["text"])
        assert short_delta == 1
        assert long_delta == 5

    def test_manipulate_classes_is_deterministic_for_seed(self) -> None:
        """Manipulation should remain reproducible with the configured data seed."""
        cfg = Config(text_field_separator=" | ")
        source = _balance_df()

        first = manipulate_classes(
            cfg=cfg,
            df=source,
            text_column="text",
            label_column="label",
            target_labels=None,
            perturbation_names=["delete_random_char", "qwerty_misspell"],
            perturbation_mean=0.2,
            perturbation_variance=0.01,
            sample_fraction=1.0,
            seed=9,
        )
        second = manipulate_classes(
            cfg=cfg,
            df=source,
            text_column="text",
            label_column="label",
            target_labels=None,
            perturbation_names=["delete_random_char", "qwerty_misspell"],
            perturbation_mean=0.2,
            perturbation_variance=0.01,
            sample_fraction=1.0,
            seed=9,
        )

        assert first["text"].tolist() == second["text"].tolist()

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
            label_harmonization_enabled=False,
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

    def test_get_splits_reuses_prepared_splits_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handler should reuse cached split-level data when config metadata matches."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.parquet"
        _processed_df(num_rows=20).to_parquet(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.parquet",
            train_size=0.7,
            val_size=0.2,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
        )
        first_handler = DataHandler(cfg)
        first_handler._write_processing_metadata(
            first_handler._build_processing_metadata()
        )
        first_splits = first_handler.get_splits()

        second_handler = DataHandler(cfg)

        def fail_prepare(processed_df: pd.DataFrame) -> DataSplits:
            raise AssertionError("prepared splits cache was not reused")

        monkeypatch.setattr(
            second_handler,
            "_prepare_splits_from_processed",
            fail_prepare,
        )
        cached_splits = second_handler.get_splits()

        assert len(first_splits.train) == len(cached_splits.train)
        assert len(first_splits.val) == len(cached_splits.val)
        assert len(first_splits.test) == len(cached_splits.test)

    def test_get_splits_rebuilds_prepared_splits_when_split_config_changes(
        self, tmp_path: Path
    ) -> None:
        """Split-level cache keys should include split configuration."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.parquet"
        _processed_df(num_rows=20).to_parquet(processed_path, index=False)

        base_cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.parquet",
            train_size=0.7,
            val_size=0.2,
            test_size=0.1,
            dataset_size=1.0,
            data_sources=[],
        )
        base_handler = DataHandler(base_cfg)
        base_handler._write_processing_metadata(
            base_handler._build_processing_metadata()
        )
        base_splits = base_handler.get_splits()

        changed_cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.parquet",
            train_size=0.5,
            val_size=0.25,
            test_size=0.25,
            dataset_size=1.0,
            data_sources=[],
        )
        changed_handler = DataHandler(changed_cfg)
        changed_splits = changed_handler.get_splits()

        assert len(base_splits.train) == 14
        assert len(changed_splits.train) == 10
        assert len(changed_splits.val) == 5
        assert len(changed_splits.test) == 5

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
            balance_strategy="floor",
            balance_floor=10,
            balance_floor_decay=0.25,
            base_perturbation_rate=0.2,
        )
        handler_base = DataHandler(cfg_base)
        prior_metadata = handler_base._build_processing_metadata()
        prior_metadata["balance_strategy"] = "floor"
        prior_metadata["balance_floor"] = 10
        prior_metadata["balance_floor_decay"] = 0.25
        prior_metadata["balance_base_perturbation_rate"] = 0.2
        handler_base._write_processing_metadata(prior_metadata)

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
            balance_floor=0,
            balance_floor_decay=0.0,
            base_perturbation_rate=0.0,
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
            label_harmonization_enabled=False,
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
            max_label_count=2,
            dataset_size=1.0,
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            label_harmonization_enabled=False,
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

    def test_get_splits_reprocesses_training_input_change_for_all_splits(
        self, tmp_path: Path
    ) -> None:
        """Changing training_input should refresh text in train, validation, and test."""
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            [
                [
                    f"cause-{idx}",
                    f"A{idx:03d}",
                    "",
                    "1" if idx % 2 == 0 else "2",
                    str(20 + idx),
                    f"RID-{idx:03d}",
                ]
                for idx in range(20)
            ]
        ).to_csv(raw_path, index=False)
        processed_dir = tmp_path / "processed"
        data_sources = [
            DataSourceConfig(
                source_id="csv_source",
                path="raw.csv",
                mapping_id="test_mapping",
            )
        ]
        common_config = dict(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            data_sources=data_sources,
            dataset_size=1.0,
            train_size=0.5,
            val_size=0.25,
            test_size=0.25,
            balance_strategy="none",
            base_perturbation_rate=0.0,
            label_harmonization_enabled=False,
        )
        mapping_registry = {"test_mapping": _make_mapping()}

        initial_handler = DataHandler(
            Config(training_input=["cod", "age", "sex"], **common_config),
            mapping_registry=mapping_registry,
        )
        _ = initial_handler.get_splits()
        initial_processed = pd.read_csv(initial_handler.processed_path)
        assert initial_processed["text"].str.startswith("cod:").all()

        changed_handler = DataHandler(
            Config(training_input=["sex", "age"], **common_config),
            mapping_registry=mapping_registry,
        )
        splits = changed_handler.get_splits()

        for split in [splits.train, splits.val, splits.test]:
            assert not split.empty
            assert split["text"].str.startswith("sex:").all()
            assert split["text"].str.contains("age:").all()
            assert not split["text"].str.contains("cod:").any()

        refreshed_processed = pd.read_csv(changed_handler.processed_path)
        assert refreshed_processed["text"].str.startswith("sex:").all()
        assert not refreshed_processed["text"].str.contains("cod:").any()

    def test_get_splits_reprocesses_max_label_count_change_for_all_splits(
        self, tmp_path: Path
    ) -> None:
        """Changing max_label_count should refresh labels across train, val, and test."""
        raw_path = tmp_path / "raw.csv"
        pd.DataFrame(
            [
                [
                    f"cause-{idx}",
                    f"A{idx:03d}.000",
                    f"J{idx:03d}.100",
                    "1",
                    str(20 + idx),
                    f"RID-{idx:03d}",
                    f"R{idx:03d}.900",
                ]
                for idx in range(20)
            ]
        ).to_csv(raw_path, index=False)
        processed_dir = tmp_path / "processed"
        data_sources = [
            DataSourceConfig(
                source_id="csv_source",
                path="raw.csv",
                mapping_id="test_mapping",
            )
        ]
        common_config = dict(
            data_raw_dir=str(tmp_path),
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            data_sources=data_sources,
            training_input=["cod", "age", "sex"],
            dataset_size=1.0,
            train_size=0.5,
            val_size=0.25,
            test_size=0.25,
            balance_strategy="none",
            base_perturbation_rate=0.0,
            label_harmonization_enabled=False,
        )
        mapping_registry = {"test_mapping": _make_mapping(multi_code_cols=[2, 6])}

        initial_handler = DataHandler(
            Config(max_label_count=1, **common_config),
            mapping_registry=mapping_registry,
        )
        initial_splits = initial_handler.get_splits()
        for split in [initial_splits.train, initial_splits.val, initial_splits.test]:
            assert not split["label"].str.contains(",").any()

        changed_handler = DataHandler(
            Config(max_label_count=2, **common_config),
            mapping_registry=mapping_registry,
        )
        changed_splits = changed_handler.get_splits()

        for split in [changed_splits.train, changed_splits.val, changed_splits.test]:
            assert not split.empty
            assert split["label"].str.contains(",").all()
            assert split["y_codes"].apply(len).eq(2).all()

        refreshed_processed = pd.read_csv(changed_handler.processed_path)
        assert refreshed_processed["label"].str.contains(",").all()

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

    def test_get_splits_holds_out_configured_source_before_sampling(
        self, tmp_path: Path
    ) -> None:
        """hold_out_dataset should remove one source from train/val/test splits."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        source_df = pd.concat(
            [
                _processed_df(num_rows=20),
                _processed_df(num_rows=5).assign(
                    source_id="external",
                    record_id=lambda df: [f"EXT-{idx:03d}" for idx in range(len(df))],
                ),
            ],
            ignore_index=True,
        )
        source_df.to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            dataset_size=0.5,
            hold_out_dataset="external",
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())
        splits = handler.get_splits()

        assert len(splits.train) == 8
        assert len(splits.val) == 1
        assert len(splits.test) == 1
        assert splits.holdout is not None
        assert len(splits.holdout) == 5
        assert set(splits.train["source_id"]) == {"test_source"}
        assert set(splits.val["source_id"]) == {"test_source"}
        assert set(splits.test["source_id"]) == {"test_source"}
        assert set(splits.holdout["source_id"]) == {"external"}

    def test_get_splits_rejects_unknown_hold_out_dataset(self, tmp_path: Path) -> None:
        """Unknown hold-out source ids should fail before training starts."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        _processed_df(num_rows=10).to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            hold_out_dataset="missing_source",
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())

        with pytest.raises(ValueError, match="missing_source"):
            handler.get_splits()

    def test_get_splits_samples_holdout_eval_when_enabled(self, tmp_path: Path) -> None:
        """hold_out_evaluate_ratio should sample holdout rows for interim eval."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "training.csv"
        source_df = pd.concat(
            [
                _processed_df(num_rows=10),
                _processed_df(num_rows=20).assign(
                    source_id="external",
                    record_id=lambda df: [f"EXT-{idx:03d}" for idx in range(len(df))],
                ),
            ],
            ignore_index=True,
        )
        source_df.to_csv(processed_path, index=False)

        cfg = Config(
            data_processed_dir=str(processed_dir),
            processed_filename="training.csv",
            hold_out_dataset="external",
            hold_out_evaluate_per="epoch",
            hold_out_evaluate_ratio=0.25,
            dataset_size=1.0,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            data_sources=[],
        )
        handler = DataHandler(cfg)
        handler._write_processing_metadata(handler._build_processing_metadata())
        splits = handler.get_splits()

        assert splits.holdout is not None
        assert splits.holdout_eval is not None
        assert len(splits.holdout) == 20
        assert len(splits.holdout_eval) == 5
        assert set(splits.holdout_eval["source_id"]) == {"external"}

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
            base_perturbations=["delete_random_char"],
            base_perturbation_rate=1.0,
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
