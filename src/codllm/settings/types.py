from typing import Literal

TrainingInput = Literal["cod", "age", "sex"]
MultiCodSyntheticSourceScope = Literal["within_source", "any_source"]
BalanceStrategy = Literal["none", "floor"]
WandbMode = Literal["auto", "online", "offline", "disabled"]
WandbLogModel = Literal["false", "end", "checkpoint"]
WandbRunConfigMode = Literal["minimal", "standard", "full"]
WandbMetricMode = Literal["core", "standard", "all"]
TorchDType = Literal["auto", "float16", "bfloat16", "float32"]
EvalStrategy = Literal["no", "steps", "epoch"]
HoldOutEvaluatePer = Literal["steps", "epoch"]
SaveStrategy = Literal["no", "steps", "epoch", "best"]
ModelTask = Literal["seq2seq", "sequence_classification"]
LRSchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
]
SaveStrategyBestMetric = Literal[
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
]
