#!/usr/bin/env python

if "__main__" == __name__:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--accuracy", choices=["fp32", "bf16", "int8", "nf4", "bool"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", default="wip-line-distilbert")
    parser.add_argument("subcommand", choices=["dataset", "baseline", "fine-tune", "test"])
    parser.add_argument("--test-file", default="test.tsv")
    parser.add_argument("--train-file", default="train.tsv")
    name_space = parser.parse_args()

    if "dataset" == name_space.subcommand:

        def load_merged_jp():
            from os.path import join
            from pandas import merge
            from pandas import read_parquet

            return merge(
                on="product_id",
                *[
                    read_parquet(
                        join(
                            "esci-data",
                            "shopping_queries_dataset",
                            f"shopping_queries_dataset_{suffix}.parquet",
                        ),
                        filters=[("product_locale", "==", "jp")],
                    )
                    for suffix in ("products", "examples")
                ],
            )

        merged_jp = load_merged_jp()
        merged_jp["expect"] = merged_jp["esci_label"].apply(
            lambda esci_label: {"E": 1.0, "S": 0.0, "C": 0.0, "I": 0.0}[esci_label]
        )
        for column in ("query", "product_title"):  # For handling LF in queries and titles
            merged_jp[column] = merged_jp[column].apply(lambda s: " ".join(s.split()))
        for split in ("train", "test"):
            merged_jp[merged_jp.split == split].to_csv(
                {"train": name_space.train_file, "test": name_space.test_file}[split],
                sep="\t",
                columns=["expect", "query", "product_title"],
                index=False,
            )

    def get_sentence_transformer(model_name_or_path):
        if name_space.accuracy not in ("fp32", "bool"):
            assert name_space.subcommand in ("baseline", "test")
        if "bf16" == name_space.accuracy:
            import torch
            torch.set_default_dtype(torch.bfloat16)
            name_space.batch_size *= 2
        if "int8" == name_space.accuracy:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            name_space.batch_size *= 4
        if "nf4" == name_space.accuracy:
            from transformers import BitsAndBytesConfig
            import torch
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            name_space.batch_size *= 2

        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Pooling
        from sentence_transformers.models import Transformer

        transformer_module = Transformer(
            model_name_or_path,
            tokenizer_args={"trust_remote_code": True},
            # model_args={"quantization_config": bnb_config},  # NOT effective?
        )
        if name_space.accuracy in ("int8", "nf4"):  # Quick-and-dirty fix
            from transformers import AutoModel
            transformer_module.auto_model = AutoModel.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
            )
            print(transformer_module.auto_model)
        pooling_module = Pooling(transformer_module.get_word_embedding_dimension())
        modules = [transformer_module, pooling_module]
        if "bool" == name_space.accuracy:
            from Hashing import Hashing
            import torch
            if "fine-tune" == name_space.subcommand:
                modules.append(Hashing(torch.nn.Tanh()))
            else:
                modules.append(Hashing(torch.sign))
        return SentenceTransformer(modules=modules)

    def test(model):
        from pandas import read_csv

        def encode(values):
            unique_values = sorted(set(values))
            unique_values = {
                string: embedding
                for string, embedding in zip(
                    unique_values,
                    model.encode(unique_values, batch_size=name_space.batch_size, show_progress_bar=True),
                )
            }
            return [unique_values[string] for string in values]

        merged_jp_test = read_csv(name_space.test_file, sep="\t")
        queries = encode(merged_jp_test["query"])
        product_titles = encode(merged_jp_test["product_title"])

        from MRR import calc_mrr
        from numpy import dot
        from numpy.linalg import norm

        actuals = []
        for query, product_title in zip(queries, product_titles):
            actuals.append(dot(query, product_title) / (norm(query) * norm(product_title)))
        merged_jp_test["actual"] = actuals
        return calc_mrr(merged_jp_test)

    if "baseline" == name_space.subcommand:
        model = get_sentence_transformer("line-corporation/line-distilbert-base-japanese")
        print(test(model))  # For example: 0.818

    if "fine-tune" == name_space.subcommand:  # takes 10 minutes with an RTX 4080
        # Ref: https://www.sbert.net/docs/training/overview.html
        from sentence_transformers import InputExample
        from sentence_transformers.losses import CosineSimilarityLoss
        from torch.utils.data import DataLoader

        from pandas import read_csv

        merged_jp_train = read_csv(name_space.train_file, sep="\t")

        train_examples = []
        for expect, query, product_title in zip(
            merged_jp_train["expect"],
            merged_jp_train["query"],
            merged_jp_train["product_title"],
        ):
            train_examples.append(InputExample(label=expect, texts=[query, product_title]))

        model = get_sentence_transformer("line-corporation/line-distilbert-base-japanese")
        model.fit(
            train_objectives=[
                (
                    DataLoader(
                        train_examples,
                        shuffle=True,
                        batch_size=name_space.batch_size,
                        num_workers=2,
                    ),
                    CosineSimilarityLoss(model),
                )
            ],
            epochs=1,
            warmup_steps=100,
            output_path=name_space.model_dir,
        )

    if "test" == name_space.subcommand:
        # Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html
        model = get_sentence_transformer(name_space.model_dir)
        print(test(model))  # For example: 0.868
