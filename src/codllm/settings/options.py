from codllm.settings.types import (
    LRSchedulerType,
    ModelTask,
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
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
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
