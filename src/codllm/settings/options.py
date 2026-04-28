from codllm.settings.types import (
    LRSchedulerType,
    ModelTask,
    MultiCodSyntheticSourceScope,
    SaveStrategyBestMetric,
    TrainingInput,
)

SUPPORTED_TRAINING_INPUTS: tuple[TrainingInput, ...] = (
    "cod",
    "age",
    "sex",
)
SUPPORTED_SAVE_STRATEGY_BEST_METRICS: tuple[SaveStrategyBestMetric, ...] = (
    "loss",
    "accuracy",
    "exact_match",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "micro_jaccard",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "sample_precision",
    "sample_recall",
    "sample_f1",
    "sample_jaccard",
    "hamming_loss",
    "hamming_score",
)
SUPPORTED_LR_SCHEDULER_TYPES: tuple[LRSchedulerType, ...] = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
)
SUPPORTED_MODEL_TASKS: tuple[ModelTask, ...] = (
    "seq2seq",
    "sequence_classification",
)
SUPPORTED_MULTICOD_SYNTHETIC_SOURCE_SCOPES: tuple[MultiCodSyntheticSourceScope, ...] = (
    "within_source",
    "any_source",
)
