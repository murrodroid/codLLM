import pandas as pd

from codllm.data_handler import DatasetMapping, _build_x, _build_y, load_dataset


def _make_mapping(**overrides) -> DatasetMapping:
    """Create a minimal mapping with sensible defaults, overriding as needed."""
    defaults = dict(
        text_col=0,
        single_code_col=1,
    )
    defaults.update(overrides)
    return DatasetMapping(**defaults)


def _row(*values) -> pd.Series:
    """Build a pd.Series from positional values."""
    return pd.Series(values)


class TestBuildX:
    def test_text_only(self):
        mapping = _make_mapping()
        row = _row("pneumonia", "A01")
        result = _build_x(row, mapping)
        assert result == "gender: unknown | year: unknown | age: unknown | text: pneumonia"

    def test_all_metadata(self):
        mapping = _make_mapping(
            text_col=3,
            gender_col=0,
            gender_map={"1": "male"},
            year_col=1,
            age_col=2,
        )
        row = _row("1", "1920", "45.5", "cholera")
        result = _build_x(row, mapping)
        assert result == "gender: male | year: 1920 | age: 45.5 | text: cholera"

    def test_missing_gender_value(self):
        mapping = _make_mapping(gender_col=1, gender_map={"1": "male"})
        row = _row("text", "9")
        result = _build_x(row, mapping)
        assert "gender: unknown" in result

    def test_missing_age(self):
        mapping = _make_mapping(age_col=1)
        row = _row("text", None)
        result = _build_x(row, mapping)
        assert "age: unknown" in result

    def test_missing_year(self):
        mapping = _make_mapping(year_col=1)
        row = _row("text", None)
        result = _build_x(row, mapping)
        assert "year: unknown" in result

    def test_missing_text(self):
        mapping = _make_mapping()
        row = _row(None, "A01")
        result = _build_x(row, mapping)
        assert "text: unknown" in result

    def test_empty_text(self):
        mapping = _make_mapping()
        row = _row("  ", "A01")
        result = _build_x(row, mapping)
        assert "text: unknown" in result


class TestBuildY:
    def test_single_code(self):
        mapping = _make_mapping()
        row = _row("text", "A01.000")
        assert _build_y(row, mapping) == ["A01.000"]

    def test_multi_codes(self):
        mapping = _make_mapping(multi_code_cols=[2, 3])
        row = _row("text", "A01", "P95.001", "J18.900")
        assert _build_y(row, mapping) == ["P95.001", "J18.900"]

    def test_multi_codes_take_precedence(self):
        mapping = _make_mapping(multi_code_cols=[2])
        row = _row("text", "A01", "P95.001")
        result = _build_y(row, mapping)
        assert result == ["P95.001"]
        assert "A01" not in result

    def test_falls_back_to_single_when_multi_empty(self):
        mapping = _make_mapping(multi_code_cols=[2])
        row = _row("text", "A01", None)
        assert _build_y(row, mapping) == ["A01"]

    def test_no_codes(self):
        mapping = _make_mapping()
        row = _row("text", None)
        assert _build_y(row, mapping) == []


class TestLoadDataset:
    def test_load_csv(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame({"text": ["cholera", "typhus"], "code": ["A00", "A01"]})
        df.to_csv(csv_path, index=False)

        mapping = _make_mapping()
        result = load_dataset(csv_path, mapping)
        assert list(result.columns) == ["X", "y"]
        assert len(result) == 2
        assert "cholera" in result.iloc[0]["X"]
        assert result.iloc[0]["y"] == ["A00"]

    def test_skip_rows(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame({"text": ["DD_001", "cholera", "typhus"], "code": ["DD_002", "A00", "A01"]})
        df.to_csv(csv_path, index=False)

        mapping = _make_mapping(skip_rows=[0])
        result = load_dataset(csv_path, mapping)
        assert len(result) == 2
        assert "cholera" in result.iloc[0]["X"]

    def test_load_xlsx(self, tmp_path):
        xlsx_path = str(tmp_path / "test.xlsx")
        df = pd.DataFrame({"text": ["cholera"], "code": ["A00"]})
        df.to_excel(xlsx_path, index=False)

        mapping = _make_mapping()
        result = load_dataset(xlsx_path, mapping)
        assert len(result) == 1
        assert result.iloc[0]["y"] == ["A00"]
