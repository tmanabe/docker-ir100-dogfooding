#!/usr/bin/env python

SEP, UNK, CLS, PAD, MASK = "[SEP]", "[UNK]", "[CLS]", "[PAD]", "[MASK]"
MAX_LENGTH = 128

if "__main__" == __name__:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "subcommand",
        choices=[
            "clone",
            "dataset",
            "tokenizer",
            "embedding-model",
            "siamese-model",
            "test",
        ],
    )
    parser.add_argument("--embedding-train-file", default="embedding-train.txt")
    parser.add_argument("--siamese-train-file", default="siamese-train.tsv")
    parser.add_argument("--test-title-file", default="test-title.txt")
    parser.add_argument("--test-query-file", default="test-query.txt")
    parser.add_argument("--ground-truth-file", default="ground-truth.txt")
    parser.add_argument("--tokenizer-file", default="ir100-dogfooding-embedding/tokenizer.json")
    parser.add_argument(
        "--tokenizer-config-file",
        default="ir100-dogfooding-embedding/tokenizer_config.json",
    )
    parser.add_argument("--embedding-model-dir", default="ir100-dogfooding-embedding")
    parser.add_argument("--siamese-model-dir", default="ir100-dogfooding-siamese")
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

        ESCI_LABEL_TO_COSINE_SIMILARITY = {
            "E": "1.0",
            "S": "1.0",
            "C": "1.0",
            "I": "-1.0",
        }

        with open(args.siamese_train_file, "w") as stf, open(args.embedding_train_file, "w") as etf:
            for esci_label, query, product_title in zip(
                merged_us_train["esci_label"],
                merged_us_train["query"],
                merged_us_train["product_title"],
            ):
                cosine_similarity = ESCI_LABEL_TO_COSINE_SIMILARITY[esci_label]
                query = " ".join(query.split())  # For handling LF in queries and titles
                product_title = " ".join(product_title.split())
                print("\t".join([cosine_similarity, query, product_title]), file=stf)
                print(query, file=etf)
                print(product_title, file=etf)

        with open(args.ground_truth_file, "w") as gtf, open(args.test_query_file, "w") as tqf, open(
            args.test_title_file, "w"
        ) as ttf:
            for esci_label, query, product_title in zip(
                merged_us_test["esci_label"],
                merged_us_test["query"],
                merged_us_test["product_title"],
            ):
                cosine_similarity = ESCI_LABEL_TO_COSINE_SIMILARITY[esci_label]
                query = " ".join(query.split())
                product_title = " ".join(product_title.split())
                print(cosine_similarity, file=gtf)
                print(query, file=tqf)
                print(product_title, file=ttf)

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
        tokenizer.train(files=[args.embedding_train_file], trainer=tokenizer_trainer)
        tokenizer.save(args.tokenizer_file)

        # For SentenceTransformer (and AutoTokenizer)
        from json import dump

        with open(args.tokenizer_config_file, "w") as tcf:
            dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "bos_token": CLS,
                    "eos_token": SEP,
                    "unk_token": UNK,
                    "sep_token": SEP,
                    "pad_token": PAD,
                    "cls_token": CLS,
                    "mask_token": MASK,
                    "model_max_length": MAX_LENGTH,
                },
                tcf,
                indent=2,
            )

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

    if "embedding-model" == args.subcommand:  # takes 10 minutes with an RTX 4080
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
            data_files={"train": [args.embedding_train_file]},
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
                output_dir=args.embedding_model_dir,
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
        trainer.save_model(args.embedding_model_dir)

    if "siamese-model" == args.subcommand:
        # Ref: https://www.sbert.net/docs/training/overview.html
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer
        from sentence_transformers.models import Pooling

        word_embedding_model = Transformer(args.embedding_model_dir, max_seq_length=MAX_LENGTH)
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        train_examples = []
        with open(args.siamese_train_file) as stf:
            for l in stf:
                cosine_similarity, query, product_title = l.split("\t")
                cosine_similarity = float(cosine_similarity)
                input_example = InputExample(label=cosine_similarity, texts=[query, product_title])
                train_examples.append(input_example)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=256)

        from sentence_transformers.losses import CosineSimilarityLoss

        train_loss = CosineSimilarityLoss(model)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            output_path=args.siamese_model_dir,
        )

    if "test" == args.subcommand:
        # Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.siamese_model_dir, device="cuda")

        with open(args.test_title_file) as ttf:
            titles = ttf.read().splitlines()
        titles = model.encode(titles, batch_size=256, show_progress_bar=True)

        with open(args.test_query_file) as tqf:
            queries = tqf.read().splitlines()
        unique_queries = list(set(queries))
        unique_queries = {
            string: embedding
            for string, embedding in zip(
                unique_queries,
                model.encode(unique_queries, batch_size=256, show_progress_bar=True),
            )
        }
        queries = [unique_queries[string] for string in queries]

        from numpy import dot
        from numpy.linalg import norm
        from tqdm.auto import tqdm

        with open(args.ground_truth_file) as gtf:
            expects = gtf.read().splitlines()
        labels = sorted(set(expects))
        counts = {label: 0 for label in labels}
        total_actuals = {label: 0.0 for label in labels}

        for title, query, expect in tqdm(zip(titles, queries, expects)):
            actual = dot(title, query) / (norm(title) * norm(query))
            counts[expect] += 1
            total_actuals[expect] += actual

        for label in labels:
            print(f"{label}: {total_actuals[label] / counts[label]}")

        # For example,
        # -1.0: 0.7077117343293025
        #  1.0: 0.849824585044569
