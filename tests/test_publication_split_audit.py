"""Offline split-support audit tests without model training or archival inputs."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from invoke import Context, Exit

import tasks
from codllm.config import Config, config_from_env
from codllm.data import DataHandler
from codllm.evaluation import split_audit
from codllm.evaluation.artifacts import persist_training_contract
from codllm.evaluation.provenance import annotate_provenance
from codllm.evaluation.reference import build_reference
from codllm.experiments.specs import load_experiment_spec


def _config(tmp_path: Path) -> Config:
    """Isolate data paths and disable all optional external data loaders."""
    cfg = Config(
        publication_eval_enabled=True,
        evaluation_protocol="cod",
        evaluation_drop_missing_cod=True,
        train_excluded_source_ids=["historic_strings_en_2024"],
        data_sources=[],
        data_processed_dir=str(tmp_path / "processed"),
        data_raw_dir=str(tmp_path / "raw"),
        label_harmonization_enabled=False,
        label_standardization_enabled=False,
        pretrain_enabled=False,
        masterlist_inject_enabled=False,
        balance_strategy="floor",
        balance_floor=0,
        base_perturbation_rate=0,
        multicod_synthetic_ratio=0,
        multicod_shuffle_labels=True,
        max_label_count=2,
        dataset_size=1,
        seed=777,
        data_seed=777,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )
    cfg.wandb.enabled = False
    return cfg


def _rows(cfg: Config, count: int = 100) -> pd.DataFrame:
    """Create two Dutch archives with shared COD groups plus excluded and missing rows."""
    rows = [
        {
            "source_id": "amsterdam_1854_1926" if i % 2 else "belgium_1920_1930",
            "record_id": str(i),
            "text": f"private-fixture-cod-{i // 2}",
            "label": "A00.000,B00.000" if i % 5 == 0 else "A00.000",
        }
        for i in range(count)
    ]
    rows += [
        {
            "source_id": "historic_strings_en_2024",
            "record_id": "reference",
            "text": "reference-only",
            "label": "C00.000",
        },
        {
            "source_id": "amsterdam_1854_1926",
            "record_id": "missing",
            "text": "",
            "label": "A00.000",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["y_codes"] = frame["label"].str.split(cfg.label_separator)
    return annotate_provenance(frame, cfg)


def _install_processed(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, data: pd.DataFrame
) -> None:
    """Supply an existing isolated processed cache while preserving its real metadata logic."""
    handler = DataHandler(cfg)
    handler.processed_path.parent.mkdir(parents=True)
    data.to_parquet(handler.processed_path, index=False)
    handler._write_processing_metadata(handler._build_processing_metadata())
    monkeypatch.setattr(DataHandler, "ensure_processed", lambda self: data.copy())


def test_audit_writes_private_support_and_never_builds_augmented_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All recipe cells share one original preparation and export counts without raw descriptions."""
    cfg = _config(tmp_path)
    data = _rows(cfg)
    _install_processed(monkeypatch, cfg, data)
    calls = []
    original = DataHandler.prepare_original_splits

    def prepare(handler: DataHandler, frame: pd.DataFrame) -> tuple:
        """Count the original-split construction calls."""
        calls.append(len(frame))
        return original(handler, frame)

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if audit tries to construct augmented training or pretraining data."""
        pytest.fail("Audit must not build augmented data or invoke training.")

    monkeypatch.setattr(DataHandler, "prepare_original_splits", prepare)
    monkeypatch.setattr(DataHandler, "get_splits", forbidden)
    monkeypatch.setattr(DataHandler, "_apply_balance_policy", forbidden)
    monkeypatch.setattr(DataHandler, "get_pretraining_train_dataframe", forbidden)
    configs = []
    for floor in (0, 450):
        for ratio in (0.3, 0.6):
            cell = copy.deepcopy(cfg)
            cell.balance_floor, cell.multicod_synthetic_ratio = floor, ratio
            configs.append(cell)
    output = tmp_path / "audit.json"
    report = split_audit.audit_splits(configs, [str(i) for i in range(4)], str(output))
    assert calls == [102]
    assert report["status"] == "review_required"
    assert not report["scientific_approval"] and not report["model_scores_computed"]
    assert all(check["passed"] for check in report["checks"])
    assert report["shared_original_partition_count"] == 1
    assert len(report["expected_prepared_caches"]) == 4
    assert all(
        cache["status"] == "not_built" for cache in report["expected_prepared_caches"]
    )
    stages = {item["stage"]: item["rows"] for item in report["preparation_ledger"]}
    assert stages["after_missing_cod_policy"] == 101
    assert stages["sampled_cohort"] == 100
    assert (
        sum(
            report["partitions"][part]["rows"]
            for part in ("original_train", "val", "test")
        )
        == 100
    )
    assert report["eligibility"]["val"]["overall"]["seen_cod"]["rows"] == 0
    assert not report["eligibility"]["val"]["overall"]["crosslingual"]["available"]
    assert not DataHandler(cfg).prepared_splits_root.exists()
    assert (
        json.loads(output.read_text())["partition_fingerprints"]
        == report["partition_fingerprints"]
    )
    assert "private-fixture-cod" not in output.read_text()
    assert "private-fixture-cod" not in output.with_suffix(".md").read_text()
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.with_suffix(".md").stat().st_mode & 0o777 == 0o600
    assert all(file["sha256"] for file in report["data"]["files"])


def test_shared_original_split_matches_production_and_label_order_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Audit identities match production before augmentation and ignore harmless target order changes."""
    cfg = _config(tmp_path)
    cfg.multicod_synthetic_ratio = 0.3
    cfg.balance_floor = 100
    data = _rows(cfg)
    _install_processed(monkeypatch, cfg, data)
    handler = DataHandler(cfg)
    original, _ = handler.prepare_original_splits(data)
    prepared = handler._prepare_splits_from_processed(data)
    assert len(prepared.train) > len(original.train)
    fingerprints = {}
    for name in ("original_train", "val", "test"):
        fingerprints[name] = split_audit.partition_fingerprint(
            getattr(original, name), cfg
        )
        assert fingerprints[name] == split_audit.partition_fingerprint(
            getattr(prepared, name), cfg
        )
    metadata = handler._build_prepared_splits_metadata()
    root = handler._prepared_splits_dir(metadata)
    handler._write_prepared_splits_cache(prepared, metadata, root)
    caches = split_audit._check_existing_caches([cfg], fingerprints)
    assert caches[0]["status"] == "matches"
    changed = prepared.val.copy()
    changed.loc[0, "text"] = "altered cached input"
    changed.to_parquet(root / "val.parquet", index=False)
    assert (
        split_audit._check_existing_caches([cfg], fingerprints)[0]["status"]
        == "mismatch"
    )
    (root / "metadata.json").write_text("{invalid-json")
    assert (
        split_audit._check_existing_caches([cfg], fingerprints)[0]["status"]
        == "mismatch"
    )


def test_eligibility_distinguishes_source_language_and_masterlist_support(
    tmp_path: Path,
) -> None:
    """French/Dutch assumptions are not guessed and planned English pretraining removes strict eligibility."""
    cfg = _config(tmp_path)
    original = _rows(cfg).iloc[:100]
    masterlist = annotate_provenance(
        pd.DataFrame(
            [
                {
                    "source_id": "masterlist_pretrain",
                    "record_id": "a",
                    "text": "English resource a",
                    "label": "A00.000",
                },
                {
                    "source_id": "masterlist_pretrain",
                    "record_id": "c",
                    "text": "English resource c",
                    "label": "C00.000",
                },
            ]
        ),
        cfg,
    )
    reference = build_reference(original, original, masterlist, cfg)
    target = annotate_provenance(
        pd.DataFrame(
            [
                {
                    "source_id": "ipswich_1871_1911",
                    "record_id": "e",
                    "text": "new english",
                    "label": "A00.000,C00.000",
                },
                {
                    "source_id": "copenhagen_may2025",
                    "record_id": "d",
                    "text": "new danish",
                    "label": "A00.000",
                },
                {
                    "source_id": "belgium_1920_1930",
                    "record_id": "b",
                    "text": "new flemish",
                    "label": "A00.000",
                },
                {
                    "source_id": "unreviewed",
                    "record_id": "u",
                    "text": "unknown language",
                    "label": "A00.000,D00.000",
                },
            ]
        ),
        cfg,
    ).assign(audit_group=range(4))
    result = split_audit._eligibility(target, reference, cfg)
    assert result["source/ipswich_1871_1911"]["crosslingual"]["target_occurrences"] == 1
    assert result["source/ipswich_1871_1911"]["crosslingual"]["multi_cod_rows"] == 1
    assert result["source/ipswich_1871_1911"]["strict_crosslingual"]["rows"] == 0
    assert result["source/ipswich_1871_1911"]["masterlist_only_label"][
        "per_code_target_occurrences"
    ] == {"C00.000": 1}
    assert result["source/copenhagen_may2025"]["strict_crosslingual"]["rows"] == 1
    assert result["source/belgium_1920_1930"]["crosslingual"]["rows"] == 0
    assert result["source/unreviewed"]["crosslingual"]["rows"] == 0
    assert result["source/unreviewed"]["absent_from_all_adaptation"][
        "per_code_target_occurrences"
    ] == {"D00.000": 1}
    assert result["overall"]["source_transfer"]["target_occurrences"] == 3
    target.loc[2, "label"] = "C00.000"
    assert "C00.000" not in reference.label_sources


def test_inventory_counts_groups_and_conflicting_targets(tmp_path: Path) -> None:
    """Repeated rows are dependence groups rather than extra independent rare-code observations."""
    cfg = _config(tmp_path)
    frame = _rows(cfg).iloc[:100].assign(audit_group=[i // 2 for i in range(100)])
    inventory = split_audit._inventory(frame, cfg)
    assert inventory["connected_groups"]["groups"] == 50
    assert inventory["connected_groups"]["largest_rows"] == 2
    assert inventory["normalized_cods_with_multiple_target_sets"] == 20
    assert inventory["per_code"]["B00.000"]["rows"] == 20
    assert inventory["per_code"]["A00.000"]["connected_groups"] == 50


def test_conflicting_duplicate_uid_is_integrity_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Silent training deduplication must not hide conflicting content under a reused row UID."""
    cfg = _config(tmp_path)
    data = _rows(cfg)
    data.loc[2, "row_uid"] = data.loc[0, "row_uid"]
    _install_processed(monkeypatch, cfg, data)
    report = split_audit.audit_splits([cfg], ["conflict"], str(tmp_path / "audit.json"))
    assert report["status"] == "integrity_failed"
    assert not next(
        c for c in report["checks"] if c["check"] == "duplicate_uid_content_consistent"
    )["passed"]


def test_real_phase_1a_cells_share_original_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact published eight-cell spec is accepted while changed split seeds are not."""
    spec = load_experiment_spec("runs/publication/interaction_confirmation.toml")
    configs = []
    for run in spec.expanded_runs():
        with monkeypatch.context() as context:
            for key, value in run.env_with_runtime_metadata(spec).items():
                context.setenv(key, value)
            configs.append(config_from_env())
    assert len(configs) == 8
    split_audit.validate_shared_audit_configs(configs)
    configs[-1].data_seed = 101
    with pytest.raises(ValueError, match="audit separately"):
        split_audit.validate_shared_audit_configs(configs)


@pytest.mark.parametrize("status", ["review_required", "integrity_failed"])
def test_task_uses_training_profile_paths_and_returns_failure_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    """The CPU audit inherits the HPC data paths but never submits jobs or approves a phase."""
    calls = []

    def runtime(
        profile: str, profiles: str, user: str | None, lucas: bool, elias: bool
    ) -> dict:
        """Represent profile-derived paths independently of the actual HPC installation."""
        assert profile == "h100" and lucas and not elias and user is None
        return {
            "CODLLM_DATA_RAW_DIR": str(tmp_path / "raw"),
            "CODLLM_DATA_PROCESSED_DIR": str(tmp_path / "processed"),
        }

    def audit(
        configs: list[Config], names: list[str], output: str, *, specification: str
    ) -> dict:
        """Capture task inputs without loading any data."""
        calls.append(names)
        assert len(configs) == len(names) == 8
        assert all(c.data_raw_dir == str(tmp_path / "raw") for c in configs)
        assert all(c.data_processed_dir == str(tmp_path / "processed") for c in configs)
        assert specification == "runs/publication/interaction_confirmation.toml"
        return {"status": status}

    monkeypatch.setattr(tasks, "_maintenance_runtime_env", runtime)
    monkeypatch.setattr(split_audit, "audit_splits", audit)
    if status == "integrity_failed":
        with pytest.raises(Exit) as caught:
            tasks.publication_audit_splits.body(Context(), profile="h100", lucas=True)
        assert caught.value.code == 2
    else:
        tasks.publication_audit_splits.body(Context(), profile="h100", lucas=True)
    assert len(calls) == 1


def test_audit_top_group_setting_is_display_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changing the display limit must not invalidate prepared training data."""
    cfg = _config(tmp_path)
    cfg.output_dir = str(tmp_path / "run")
    splits, _ = DataHandler(cfg).prepare_original_splits(_rows(cfg))
    reference = build_reference(splits.original_train, splits.train, None, cfg)
    contract = persist_training_contract(cfg, splits, reference) / "contract.json"
    before = contract.read_text()
    metadata = DataHandler(cfg)._build_prepared_splits_metadata()
    cfg.publication_audit_top_groups = 3
    assert DataHandler(cfg)._build_prepared_splits_metadata() == metadata
    persist_training_contract(cfg, splits, reference)
    assert contract.read_text() == before
    monkeypatch.setenv("CODLLM_PUBLICATION_AUDIT_TOP_GROUPS", "3")
    assert config_from_env().publication_audit_top_groups == 3
    monkeypatch.setenv("CODLLM_PUBLICATION_AUDIT_TOP_GROUPS", "0")
    with pytest.raises(ValueError):
        config_from_env()


def test_pretraining_resource_is_loaded_once_and_fingerprinted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive pretraining doses share planned original-masterlist exposure, not augmented PT datasets."""
    cfg = _config(tmp_path)
    cfg.pretrain_enabled = True
    cfg.pretrain_num_train_epochs = 4
    cfg.pretrain_masterlist_path = str(tmp_path / "masterlist.xlsx")
    Path(cfg.pretrain_masterlist_path).write_bytes(
        b"fixture resource; loader replaced for this test"
    )
    data = _rows(cfg)
    _install_processed(monkeypatch, cfg, data)
    masterlist = annotate_provenance(
        pd.DataFrame(
            [
                {
                    "source_id": "masterlist_pretrain",
                    "record_id": "a",
                    "text": "private masterlist input",
                    "label": "A00.000",
                }
            ]
        ),
        cfg,
    )
    calls = []

    def load(handler: DataHandler) -> pd.DataFrame:
        """Count source loads, without upsampling or Excel dependencies."""
        calls.append(handler.cfg.pretrain_num_train_epochs)
        return masterlist

    monkeypatch.setattr(DataHandler, "_load_pretraining_source", load)
    second = copy.deepcopy(cfg)
    second.pretrain_num_train_epochs = 48
    report = split_audit.audit_splits(
        [cfg, second], ["4", "48"], str(tmp_path / "audit.json")
    )
    assert calls == [4]
    assert report["masterlist"] == {
        "enabled": True,
        "rows": 1,
        "codes": 1,
        "languages": {"en": 1},
    }
    assert report["original_training_code_exposure"]["A00.000"][
        "planned_adaptation_languages"
    ] == ["en", "nl"]
    assert any(
        item["path"] == cfg.pretrain_masterlist_path and item["sha256"]
        for item in report["data"]["files"]
    )
    assert "private masterlist input" not in (tmp_path / "audit.json").read_text()


def test_overlap_rules_allow_heldout_cod_overlap_but_reject_split_leakage(
    tmp_path: Path,
) -> None:
    """A source holdout can share text with training; grouped validation cannot."""
    cfg = _config(tmp_path)
    data = _rows(cfg)
    first, second = data.iloc[[0]], data.iloc[[1]]
    checks = split_audit._overlap_checks({"original_train": first, "val": second}, cfg)
    assert not checks[-1]["passed"] and checks[-1]["shared_cods"] == 1
    checks = split_audit._overlap_checks(
        {"original_train": first, "holdout": second}, cfg
    )
    assert checks[-1]["passed"] and not checks[-1]["cod_disjoint_required"]
    cfg.evaluation_protocol = "row"
    assert split_audit._overlap_checks({"original_train": first, "val": second}, cfg)[
        -1
    ]["passed"]
    assert not split_audit._overlap_checks(
        {"original_train": first, "val": first}, cfg
    )[-1]["passed"]
