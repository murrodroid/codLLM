"""STEP 1: download the 5 LOSO base checkpoints (the real v1, ~1.19 GB), not the empty v0."""
import wandb

api = wandb.Api()
runs = {
    "amsterdam": "yxwuq3da",
    "belgium": "qzozbcay",
    "copenhagen": "j8gy70lj",
    "ipswich": "zk7olj7q",
    "madrid": "czp3f1xl",
}
for src, rid in runs.items():
    r = api.run(f"murromanden_data/codllm/{rid}")
    arts = [a for a in r.logged_artifacts() if a.type == "model"]
    # Pick the real checkpoint: the model artifact with the largest size (v0 is an empty placeholder).
    art = max(arts, key=lambda a: a.size)
    root = f"experiments/uncertainty/results/loso/ckpt_{src}"
    art.download(root=root)
    print(f"{src} -> {art.name} ({art.size/1e9:.2f} GB) into {root}", flush=True)
print("DOWNLOAD DONE", flush=True)
