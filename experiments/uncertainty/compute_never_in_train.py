"""Compute the never-in-train test slice (the true memorization-free generalization
slice) for the seed-333 split. Dumps test parquet_idx that never appear in train,
both by cause-text (cod) and by full input (cod+age+sex), so we can match ch5's
5,955 figure and recompute the honest unique-slice numbers.
"""
import json
import re

from codllm.settings.schema import Config
from codllm.data.handler import DataHandler

cfg = Config()
cfg.max_label_count = 3
cfg.dataset_size = 1.0
cfg.train_size = 0.9
cfg.val_size = 0.05
cfg.test_size = 0.05
cfg.training_input = ["cod", "age", "sex"]

h = DataHandler(cfg)
df = h.ensure_processed()
sp = h.split_dataframe(df)
tc = cfg.dataset_text_column


def cod(t):
    m = re.search(r"cod:\s*([^|]+)", str(t))
    return (m.group(1).strip().lower() if m else str(t).strip().lower())


def full(t):
    return str(t).strip().lower()


train_cod = set(cod(t) for t in sp.train[tc])
val_cod = set(cod(t) for t in sp.val[tc])
trainval_cod = train_cod | val_cod
test = sp.test

nit_cod = [int(i) for i, t in zip(test.index, test[tc]) if cod(t) not in train_cod]
nitv_cod = [int(i) for i, t in zip(test.index, test[tc]) if cod(t) not in trainval_cod]

n = len(test)
print(f"train n={len(sp.train)}  val n={len(sp.val)}  test n={n}")
print(f"never-in-TRAIN         (cod): {len(nit_cod)}  ({len(nit_cod)/n*100:.2f}%)")
print(f"never-in-TRAIN-OR-VAL  (cod): {len(nitv_cod)}  ({len(nitv_cod)/n*100:.2f}%)")
out = "experiments/uncertainty/results/seed333_fixed/never_in_train_idx.json"
json.dump({"cod": nit_cod, "trainval": nitv_cod, "test_n": n}, open(out, "w"))
print("dumped to", out)
