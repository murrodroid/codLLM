"""Module entrypoint for frozen publication evaluation."""

from codllm.evaluation.cli import main

if __name__ == "__main__":
    import os

    if os.getenv("CODLLM_JOB_COMMAND") == "publication-baseline":
        from codllm.config import config_from_env
        from codllm.evaluation.baselines import run_baseline

        run_baseline(config_from_env())
    else:
        main()
