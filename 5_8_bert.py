#!/usr/bin/env python

SEP, UNK, CLS, PAD, MASK = "[SEP]", "[UNK]", "[CLS]", "[PAD]", "[MASK]"
MAX_LENGTH = 128

if "__main__" == __name__:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "subcommand", choices=["clone", "dataset", "tokenizer", "model", "test"]
    )
    parser.add_argument("--train-file", default="train.txt")
    parser.add_argument("--test-file", default="test.txt")
    parser.add_argument("--ground-truth-file", default="ground-truth.txt")
    parser.add_argument(
        "--tokenizer-file", default="ir100-dogfooding-bert/tokenizer.json"
    )
    parser.add_argument("--model-dir", default="ir100-dogfooding-bert")
    args = parser.parse_args()

    if "clone" == args.subcommand:
        from os import system

        assert 0 == system("git clone https://github.com/amazon-science/esci-data.git")

    def load_merged_us():
        from os import path
        import pandas as pd

        products = pd.read_parquet(
            path.join(
                "esci-data",
                "shopping_queries_dataset",
                "shopping_queries_dataset_products.parquet",
            )
        )
        queries = pd.read_parquet(
            path.join(
                "esci-data",
                "shopping_queries_dataset",
                "shopping_queries_dataset_examples.parquet",
            )
        )
        products_us = products["us" == products.product_locale]
        queries_us = queries["us" == queries.product_locale]
        return pd.merge(products_us, queries_us, on="product_id")

    if "dataset" == args.subcommand:
        merged_us = load_merged_us()
        merged_us_train = merged_us["train" == merged_us.split]
        merged_us_test = merged_us["test" == merged_us.split]

        with open(args.train_file, "w") as tf:
            for esci_label, query, product_title in zip(
                merged_us_train["esci_label"],
                merged_us_train["query"],
                merged_us_train["product_title"],
            ):
                query = " ".join(query.split())  # For handling LF in queries and titles
                product_title = " ".join(product_title.split())
                print(" ".join([esci_label, query, SEP, product_title, SEP]), file=tf)

        with open(args.test_file, "w") as tf, open(args.ground_truth_file, "w") as gtf:
            for esci_label, query, product_title in zip(
                merged_us_test["esci_label"],
                merged_us_test["query"],
                merged_us_test["product_title"],
            ):
                query = " ".join(query.split())
                product_title = " ".join(product_title.split())
                print(" ".join([MASK, query, SEP, product_title, SEP]), file=tf)
                print(esci_label, file=gtf)

    if "tokenizer" == args.subcommand:
        # Ref: https://huggingface.co/robot-test/dummy-tokenizer-wordlevel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from tokenizers.trainers import WordLevelTrainer

        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer_trainer = WordLevelTrainer(
            vocab_size=30500,
            special_tokens=[SEP, UNK, CLS, PAD, MASK],
        )
        tokenizer.train(files=[args.train_file], trainer=tokenizer_trainer)
        tokenizer.save(args.tokenizer_file)

    def load_tokenizer():
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_file=args.tokenizer_file,
            bos_token=CLS,
            eos_token=SEP,
            unk_token=UNK,
            sep_token=SEP,
            pad_token=PAD,
            cls_token=CLS,
            mask_token=MASK,
            model_max_length=MAX_LENGTH,
        )

    if "model" == args.subcommand:  # takes 10 minutes with an RTX 4080
        # Ref: https://github.com/huggingface/transformers/blob/v4.18.0/examples/pytorch/language-modeling/run_mlm.py
        from datasets import load_dataset

        TEXT_COLUMN = "text"

        tokenizer = load_tokenizer()

        def tokenize_function(examples):
            return tokenizer(
                examples[TEXT_COLUMN],
                truncation=True,
                max_length=MAX_LENGTH,
                return_special_tokens_mask=True,
            )

        raw_datasets = load_dataset(
            "text",
            data_files={"train": [args.train_file]},
        )
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=[TEXT_COLUMN],
        )

        # Ref: https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb
        from transformers import BertConfig
        from transformers import BertForMaskedLM
        from transformers import DataCollatorForLanguageModeling
        from transformers import Trainer
        from transformers import TrainingArguments

        trainer = Trainer(
            model=BertForMaskedLM(  # , a tiny version of
                BertConfig(
                    hidden_size=192,
                    num_hidden_layers=3,
                    num_attention_heads=3,
                    intermediate_size=768,
                    max_position_embeddings=MAX_LENGTH,
                    type_vocab_size=1,
                )
            ),
            args=TrainingArguments(
                output_dir=args.model_dir,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=64,
                save_strategy="no",
                prediction_loss_only=True,
                optim="adamw_torch",
            ),
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
            ),
            train_dataset=tokenized_datasets["train"],
        )
        trainer.train()
        trainer.save_model(args.model_dir)

    if "test" == args.subcommand:  # takes 10 minutes with an RTX 4080
        from transformers import pipeline

        fill_mask = pipeline(
            "fill-mask",
            model=args.model_dir,
            tokenizer=load_tokenizer(),
            device=0,
        )

        from torch.utils.data import Dataset

        class ListDataset(Dataset):
            def __init__(self, l):
                self.body = l

            def __len__(self):
                return len(self.body)

            def __getitem__(self, i):
                return self.body[i]

        with open(args.test_file) as f:
            dataset = ListDataset(f.read().splitlines())

        with open(args.ground_truth_file) as f:
            ground_truth = f.read().splitlines()

        from tqdm.auto import tqdm

        def get_penalty(output):
            min_score = 1.0
            for d in output:
                if "I" == d["token_str"]:
                    return d["score"]  # Score of "I" (irrelevant) == penalty
                if d["score"] < min_score:
                    min_score = d["score"]
            return min_score  # or less

        labels = ["E", "S", "C", "I"]
        counts = {label: 0 for label in labels}
        total_penalties = {label: 0.0 for label in labels}
        for esci_label, output in tqdm(
            zip(ground_truth, fill_mask(dataset, batch_size=256)), total=len(dataset)
        ):
            counts[esci_label] += 1
            total_penalties[esci_label] += get_penalty(output)

        for label in labels:
            print(f"{label}: {total_penalties[label] / counts[label]}")

        # For example,
        # E: 0.07940401978097034
        # S: 0.08698296396494333
        # C: 0.07644840514785672
        # I: 0.09438480036770797
