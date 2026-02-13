from config import Config

def build_preprocess_fn(cfg: Config, tokenizer):
    def preprocess(batch):
        sources = batch["text"]
        targets = batch["label"]

        model_inputs = tokenizer(
            sources,
            max_length=cfg.max_source_length,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=cfg.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess