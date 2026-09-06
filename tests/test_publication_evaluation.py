"""CPU-only tests for the publication evaluation contract."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codllm.config import Config, config_from_env
from codllm.data import DataHandler
from codllm.evaluation.artifacts import PublicationReporter, persist_training_contract
from codllm.evaluation.context import evaluation_rows
from codllm.evaluation.provenance import (
    annotate_provenance,
    evaluation_records,
    normalize_cod,
)
from codllm.evaluation.reference import build_reference
from codllm.evaluation.scoring import (
    crosslingual_codes,
    reference_macro_f1,
    score_publication,
)
from codllm.evaluation.splitting import validate_group_integrity
from codllm.input.multicod import build_synthetic_multicod_rows


def config(tmp_path: Path) -> Config:
    """Build a publication config with all local data/model side effects isolated."""
    cfg = Config(
        publication_eval_enabled=True,
        evaluation_protocol="cod",
        prediction_export_enabled=True,
        train_excluded_source_ids=["historic_strings_en_2024"],
        final_test_eval_enabled=False,
        label_harmonization_enabled=False,
        label_standardization_enabled=False,
        dataset_size=1,
        max_label_count=2,
        base_perturbation_rate=0,
        output_dir=str(tmp_path / "run"),
        data_processed_dir=str(tmp_path / "processed"),
        seed=777,
        data_seed=777,
    )
    cfg.wandb.enabled = False
    return cfg


def frame(cfg: Config, count: int = 80) -> pd.DataFrame:
    """Build grouped repeated descriptions spanning two same-language archives."""
    return annotate_provenance(
        pd.DataFrame(
            {
                "source_id": [
                    "amsterdam_1854_1926" if i % 2 else "belgium_1920_1930"
                    for i in range(count)
                ],
                "record_id": [str(i) for i in range(count)],
                "text": [f"description {i // 2}" for i in range(count)],
                "label": ["A00.000" if i % 3 else "B00.000" for i in range(count)],
                "y_codes": [
                    ["A00.000"] if i % 3 else ["B00.000"] for i in range(count)
                ],
            }
        ),
        cfg,
    )


def test_cod_normalization_ignores_metadata_not_diacritics(tmp_path: Path) -> None:
    """COD identity is full-text, punctuation-preserving, and independent of formatted metadata."""
    cfg = config(tmp_path)
    cfg.training_input = ["cod", "age", "sex"]
    raw = pd.DataFrame(
        {
            "source_id": ["ipswich_1871_1911"] * 2,
            "record_id": ["a", "b"],
            "text": [
                "cod: FéVer  | age: 7 | sex: f",
                "cod: FE\u0301VER | age: 99 | sex: m",
            ],
            "label": ["A00.000"] * 2,
        }
    )
    annotated = annotate_provenance(raw, cfg)
    assert annotated["cod_key"].tolist() == ["féver", "féver"]
    assert normalize_cod("fever") != normalize_cod("féver")
    assert normalize_cod("a.b") != normalize_cod("a b")


def test_global_group_split_and_linked_records(tmp_path: Path) -> None:
    """Both COD duplicates across archives and linked records stay inside one partition."""
    cfg = config(tmp_path)
    data = frame(cfg)
    data.loc[2, ["record_id", "source_id"]] = data.loc[
        0, ["record_id", "source_id"]
    ].to_numpy()
    splits = DataHandler(cfg).split_dataframe(data)
    validate_group_integrity(splits, cfg)
    again = DataHandler(cfg).split_dataframe(data)
    assert splits.train["row_uid"].tolist() == again.train["row_uid"].tolist()
    assert sum(len(part) for part in (splits.train, splits.val, splits.test)) == len(
        data
    )
    cfg.data_seed = 101
    changed = DataHandler(cfg).split_dataframe(data)
    assert set(changed.train["row_uid"]) != set(splits.train["row_uid"])


def test_flemish_is_same_language_transfer_and_masterlist_exposure(
    tmp_path: Path,
) -> None:
    """Belgian Dutch is not a separate language; English pretraining invalidates strict English transfer."""
    cfg = config(tmp_path)
    original = frame(cfg)
    masterlist = annotate_provenance(
        pd.DataFrame(
            {
                "source_id": ["masterlist_pretrain"],
                "record_id": ["m"],
                "text": ["catalogue disease"],
                "label": ["A00.000"],
            }
        ),
        cfg,
    )
    reference = build_reference(original, original, masterlist, cfg)
    assert set(original["language"]) == {"nl"}
    assert not crosslingual_codes({"A00.000"}, "nl", reference)
    assert crosslingual_codes({"A00.000"}, "en", reference) == {"A00.000"}
    assert not crosslingual_codes({"A00.000"}, "en", reference, strict_adaptation=True)
    assert crosslingual_codes({"A00.000"}, "da", reference, strict_adaptation=True) == {
        "A00.000"
    }
    reference.label_languages["A00.000"].add("und")
    assert not crosslingual_codes({"A00.000"}, "da", reference)


def test_multicod_slice_preserves_unrelated_false_positives(tmp_path: Path) -> None:
    """Eligible-target recall must not be confused with full-set exact accuracy."""
    cfg = config(tmp_path)
    original = frame(cfg)
    reference = build_reference(original, original, None, cfg)
    target = annotate_provenance(
        pd.DataFrame(
            {
                "source_id": ["copenhagen_may2025"],
                "record_id": ["target"],
                "text": ["new text"],
                "label": ["A00.000"],
            }
        ),
        cfg,
    )
    metrics, records = score_publication(
        [{"A00.000", "Z99.999"}],
        [{"A00.000"}],
        evaluation_records(target, cfg),
        reference,
    )
    assert metrics["pub_v1_crosslingual_eligible_recall"] == 1
    assert metrics["pub_v1_crosslingual_accuracy"] == 0
    assert metrics["pub_v1_unseen_cod_row_count"] == 1
    assert records[0]["crosslingual_targets"] == ["A00.000"]
    assert np.isnan(metrics["pub_v1_historically_unseen_label_accuracy"])
    assert reference_macro_f1([{"A00.000", "Z99.999"}], [{"A00.000"}]) == 1


def test_synthetic_rows_record_all_constituents(tmp_path: Path) -> None:
    """Synthetic identities are distinct and preserve training-only constituent provenance."""
    cfg = config(tmp_path)
    cfg.multicod_synthetic_ratio = 0.5
    original = frame(cfg)
    synthetic = build_synthetic_multicod_rows(original, cfg)
    assert not synthetic.empty
    assert not set(synthetic["row_uid"]) & set(original["row_uid"])
    assert set(synthetic["language_candidates"]) == {"nl"}
    for value in synthetic["synthetic_parent_uids"]:
        assert set(json.loads(value)) <= set(original["row_uid"])


def test_cache_identity_includes_protocol_and_language_curation(tmp_path: Path) -> None:
    """Protocol and reviewed-language changes invalidate prepared split caches."""
    cfg = config(tmp_path)
    handler = DataHandler(cfg)
    initial = handler._build_prepared_splits_metadata()
    cfg.evaluation_protocol = "row"
    assert handler._build_prepared_splits_metadata() != initial
    cfg.evaluation_protocol = "cod"
    path = tmp_path / "languages.toml"
    path.write_text('version = 1\n[sources.test]\nlanguages = ["nl"]\n')
    cfg.evaluation_language_metadata_path = str(path)
    first = handler._build_prepared_splits_metadata()
    path.write_text('version = 1\n[sources.test]\nlanguages = ["da"]\n')
    assert handler._build_prepared_splits_metadata() != first


def test_export_is_local_aligned_and_resume_contract_is_frozen(tmp_path: Path) -> None:
    """Selected predictions survive logging failures and incompatible resumes are rejected."""
    cfg = config(tmp_path)
    splits = DataHandler(cfg).split_dataframe(frame(cfg))
    splits.original_train = splits.train.copy()
    reference = build_reference(splits.train, splits.train, None, cfg)
    persist_training_contract(cfg, splits, reference)
    reporter = PublicationReporter(cfg, reference)
    reporter.exporting = True
    token = evaluation_rows.set(evaluation_records(splits.val, cfg))
    try:
        labels = [{label} for label in splits.val["label"]]
        reporter(labels, labels)
    finally:
        evaluation_rows.reset(token)
    summary_path = next((Path(cfg.output_dir) / "publication").rglob("selected.json"))
    summary = json.loads(summary_path.read_text())
    predictions = pd.read_parquet(summary_path.parent / summary["predictions"])
    assert predictions["row_uid"].tolist() == splits.val["row_uid"].tolist()
    assert "cod_text" not in predictions
    assert summary["metrics"]["pub_v1_accuracy"] == 1
    splits.val.loc[0, "row_uid"] = "different-row"
    with pytest.raises(ValueError, match="manifests changed"):
        persist_training_contract(cfg, splits, reference)


def test_new_configuration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """New publication fields resolve through the same configuration layer as every TOML."""
    monkeypatch.setenv("CODLLM_EVALUATION_PROTOCOL", "cod")
    monkeypatch.setenv("CODLLM_PUBLICATION_EVAL_ENABLED", "true")
    monkeypatch.setenv("CODLLM_PROCESSED_FILENAME", "v1.parquet")
    monkeypatch.setenv("CODLLM_TRAIN_SAMPLE_FRACTION", ".4")
    cfg = config_from_env()
    assert cfg.evaluation_protocol == "cod" and cfg.publication_eval_enabled
    assert cfg.processed_filename == "v1.parquet" and cfg.train_sample_fraction == 0.4
    monkeypatch.setenv("CODLLM_EVALUATION_PROTOCOL", "invented")
    with pytest.raises(ValueError, match="row or cod"):
        config_from_env()


def test_tiny_cpu_training_exports_selected_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a real Trainer/model/tokenizer without downloads, GPUs, or W&B."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        PreTrainedTokenizerFast,
        T5Config,
        T5ForConditionalGeneration,
    )

    from codllm.training import pipeline

    cfg = config(tmp_path)
    cfg.device = torch.device("cpu")
    cfg.device_map = None
    cfg.dataloader_num_workers = 0
    cfg.num_train_epochs = 1
    cfg.per_device_train_batch_size = 8
    cfg.per_device_eval_batch_size = 8
    cfg.gradient_accumulation_steps = 1
    cfg.save_strategy = "best"
    cfg.save_strategy_best_metric = "macro_f1"
    cfg.load_best_model_at_end = True
    cfg.max_target_length = 8
    cfg.max_label_count = 1
    cfg.label_code_length = 1
    cfg.max_target_length_buffer = 0
    cfg.uncertainty_eval = False
    data = frame(cfg, 40)
    splits = DataHandler(cfg).split_dataframe(data)
    splits.original_train = splits.train.copy()

    class FixtureHandler:
        """Return deterministic original data without accessing raw project files."""

        def get_splits(self, force_reprocess: bool = False):
            """Return the fixture split contract."""
            return splits

    vocab = {
        "<pad>": 0,
        "</s>": 1,
        "<unk>": 2,
        "description": 3,
        "A00.000": 4,
        "B00.000": 5,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=6,
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    monkeypatch.setattr(
        pipeline,
        "initialize_training_components",
        lambda **kwargs: (model, tokenizer, True),
    )
    trainer, _, _ = pipeline.train(cfg, data_handler=FixtureHandler())
    assert trainer.args.device.type == "cpu"
    assert trainer.state.best_model_checkpoint
    selected = list(
        (Path(cfg.output_dir) / "publication" / "predictions").rglob("selected.json")
    )
    assert len(selected) == 1
    summary = json.loads(selected[0].read_text())
    assert (
        summary["identity"]["best_model_checkpoint"]
        == trainer.state.best_model_checkpoint
    )
    assert summary["metrics"]["pub_v1_row_count"] == len(splits.val)
    assert not list(
        (Path(cfg.output_dir) / "publication" / "predictions" / "test").glob("*")
    )
    from codllm.evaluation.cli import evaluate_checkpoint

    cfg.evaluation_checkpoint = trainer.state.best_model_checkpoint
    cfg.evaluation_reference_dir = str(Path(cfg.output_dir) / "publication")
    cfg.evaluation_scope = "val"
    cfg.output_dir = str(tmp_path / "frozen")
    metrics = evaluate_checkpoint(cfg)
    assert metrics["pub_v1_accuracy"] == summary["metrics"]["pub_v1_accuracy"]
    assert (Path(cfg.output_dir) / "publication" / "frozen_evaluation.json").exists()


@pytest.mark.parametrize("method", ["lookup", "linear", "retrieval"])
def test_baselines_use_original_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """All baseline implementations fit and export without models, network access, or held-out fitting."""
    from codllm.evaluation.baselines import run_baseline

    cfg = config(tmp_path)
    cfg.publication_baseline = method
    cfg.baseline_max_iter = 3
    cfg.baseline_max_features = 100
    cfg.baseline_thresholds = [0.2, 0.5]
    data = frame(cfg)
    splits = DataHandler(cfg).split_dataframe(data)
    splits.original_train = splits.train.copy()
    monkeypatch.setattr(DataHandler, "get_splits", lambda self: splits)
    monkeypatch.setattr(
        DataHandler, "_load_masterlist_source", lambda *args: data.iloc[:4].copy()
    )
    results = run_baseline(cfg)
    assert results["val"]["pub_v1_row_count"] == len(splits.val)
    assert "test" not in results
    assert (Path(cfg.output_dir) / "publication" / "baseline.joblib").exists()


def test_phase_gate_tracks_candidate_edits(tmp_path: Path) -> None:
    """Later stages cannot run with absent approvals or edited candidate settings."""
    import shutil

    from codllm.evaluation.workflow import approve_stage, validate_publication_config

    root = tmp_path / "runs"
    shutil.copytree("runs", root)
    cfg = config(tmp_path)
    cfg.publication_gate = "source_pair"
    cfg.publication_decisions_path = str(root / "publication" / "decisions.json")
    with pytest.raises(ValueError, match="not reviewed"):
        validate_publication_config(cfg)
    approve_stage("source_pair", cfg.publication_decisions_path, "Fixture review")
    validate_publication_config(cfg)
    candidate = root / "publication" / "candidate_winner.toml"
    candidate.write_text(
        candidate.read_text().replace(
            "CODLLM_BALANCE_FLOOR = 0", "CODLLM_BALANCE_FLOOR = 450"
        )
    )
    with pytest.raises(ValueError, match="stale"):
        validate_publication_config(cfg)


def test_frozen_manifest_rejects_tampering(tmp_path: Path) -> None:
    """Evaluation uses the exact recorded partition, not a subsequently edited parquet."""
    from codllm.evaluation.artifacts import load_manifest

    cfg = config(tmp_path)
    splits = DataHandler(cfg).split_dataframe(frame(cfg))
    splits.original_train = splits.train.copy()
    directory = persist_training_contract(
        cfg, splits, build_reference(splits.train, splits.train, None, cfg)
    )
    assert len(load_manifest(directory, "val")) == len(splits.val)
    path = next((directory / "manifests").glob("val-*.parquet"))
    frame_data = pd.read_parquet(path)
    frame_data.loc[0, "label"] = "Z99.999"
    frame_data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="modified"):
        load_manifest(directory, "val")


def test_source_fold_and_fixed_sample_seed(tmp_path: Path) -> None:
    """Cohort sampling is fixed while grouped split seeds vary; held-out rows never contribute synthesis."""
    cfg = config(tmp_path)
    cfg.dataset_size = 0.5
    cfg.dataset_sample_seed = 777
    cfg.hold_out_dataset = "belgium_1920_1930"
    cfg.multicod_synthetic_ratio = 0.3
    cfg.balance_strategy = "none"
    data = frame(cfg, 300)
    cfg.train_excluded_source_ids = []
    first = DataHandler(cfg)._prepare_splits_from_processed(data)
    validate_group_integrity(first, cfg)
    cfg.data_seed = 101
    second = DataHandler(cfg)._prepare_splits_from_processed(data)
    validate_group_integrity(second, cfg)
    first_cohort = set(
        pd.concat([first.original_train, first.val, first.test])["row_uid"]
    )
    second_cohort = set(
        pd.concat([second.original_train, second.val, second.test])["row_uid"]
    )
    assert first_cohort == second_cohort
    assert set(first.original_train["row_uid"]) != set(second.original_train["row_uid"])
    assert set(first.holdout["row_uid"]) == set(second.holdout["row_uid"])


@pytest.mark.parametrize("finetune_exists", [False, True])
def test_pretraining_resume_keeps_stage_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finetune_exists: bool
) -> None:
    """A partial pretraining slot must not fine-tune; a fine-tuning resume must not repeat pretraining."""
    from types import SimpleNamespace

    from codllm.training import pipeline
    from codllm.training.stages import build_finetune_stage, build_pretraining_stage

    cfg = config(tmp_path)
    cfg.auto_resume = True
    calls = []
    trainer = SimpleNamespace(model=object())

    def stage_run(**kwargs):
        """Record which stage would execute without starting a trainer."""
        calls.append(kwargs["stage"].name)
        return trainer

    monkeypatch.setattr(
        pipeline,
        "initialize_training_components",
        lambda **kwargs: (object(), object(), True),
    )
    monkeypatch.setattr(
        pipeline, "_has_existing_checkpoint", lambda path: finetune_exists
    )
    monkeypatch.setattr(pipeline, "run_training_stage", stage_run)
    monkeypatch.setattr(pipeline.run_markers, "is_resume_needed", lambda path: True)
    returned, _ = pipeline._train_with_pretraining(
        cfg, build_pretraining_stage(cfg), build_finetune_stage(cfg), object(), object()
    )
    assert returned is trainer
    assert calls == (["finetune"] if finetune_exists else ["pretrain"])


def test_paired_bootstrap_links_records_and_rejects_mismatched_rows(
    tmp_path: Path,
) -> None:
    """Bootstrap compares paired predictions with connected COD/record resampling units."""
    from codllm.evaluation.analysis import paired_bootstrap

    data = pd.DataFrame(
        {
            "row_uid": ["1", "2", "3", "4"],
            "record_id": ["a", "a", "b", "c"],
            "source_id": ["source"] * 4,
            "cod_hash": ["cod1", "cod2", "cod3", "cod3"],
            "labels": [["A00.000"]] * 4,
            "predictions": [["A00.000"]] * 4,
        }
    )
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    data.to_parquet(first, index=False)
    data.to_parquet(second, index=False)
    result = paired_bootstrap(
        str(first), str(second), str(tmp_path / "interval.json"), replicates=10
    )
    assert result["groups"] == 2
    assert result["ci95"] == [0, 0]
    data.iloc[:3].to_parquet(second, index=False)
    with pytest.raises(ValueError, match="exactly the same rows"):
        paired_bootstrap(
            str(first), str(second), str(tmp_path / "bad.json"), replicates=10
        )
