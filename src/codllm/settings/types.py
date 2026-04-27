from typing import Literal

TrainingInput = Literal["cod", "age", "sex"]
MultiCodSyntheticSourceScope = Literal["within_source", "any_source"]
BalanceStrategy = Literal["none", "upsample", "sqrt"]
WandbMode = Literal["auto", "online", "offline", "disabled"]
WandbLogModel = Literal["false", "end", "checkpoint"]
TorchDType = Literal["auto", "float16", "bfloat16", "float32"]
EvalStrategy = Literal["no", "steps", "epoch"]
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
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]
