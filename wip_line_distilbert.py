#!/usr/bin/env python

if "__main__" == __name__:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-dir", default="wip-line-distilbert")
    parser.add_argument(
        "subcommand",
        choices=["dataset", "baseline", "fine-tune", "test"],
    )
    parser.add_argument("--test-file", default="test.tsv")
    parser.add_argument("--train-file", default="train.tsv")
    args = parser.parse_args()

    if "dataset" == args.subcommand:

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
        merged_jp["cosine_similarity"] = merged_jp["esci_label"].apply(
            lambda esci_label: {"E": 1.0, "S": 1.0, "C": 1.0, "I": 0.0}[esci_label]
        )
        for column in ("query", "product_title"):  # For handling LF in queries and titles
            merged_jp[column] = merged_jp[column].apply(lambda s: " ".join(s.split()))
        for split in ("train", "test"):
            merged_jp[merged_jp.split == split].to_csv(
                {"train": args.train_file, "test": args.test_file}[split],
                sep="\t",
                columns=["cosine_similarity", "query", "product_title"],
                index=False,
            )

    def get_sentence_transformer(model_name_or_path):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Pooling
        from sentence_transformers.models import Transformer

        word_embedding_module = Transformer(
            model_name_or_path,
            tokenizer_args={"trust_remote_code": True},
        )
        pooling_module = Pooling(word_embedding_module.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_module, pooling_module])

    def test(model):
        from pandas import read_csv

        merged_jp_test = read_csv(args.test_file, sep="\t")

        unique_queries = sorted(set(merged_jp_test["query"]))
        unique_queries = {
            string: embedding
            for string, embedding in zip(
                unique_queries,
                model.encode(unique_queries, batch_size=args.batch_size, show_progress_bar=True),
            )
        }
        queries = [unique_queries[string] for string in merged_jp_test["query"]]

        product_titles = model.encode(
            merged_jp_test["product_title"], batch_size=args.batch_size, show_progress_bar=True
        )

        from numpy import dot
        from numpy.linalg import norm

        labels = sorted(set(merged_jp_test["cosine_similarity"]))
        totals, counts = {label: 0.0 for label in labels}, {label: 0 for label in labels}
        for cosine_similarity, query, product_title in zip(
            merged_jp_test["cosine_similarity"], queries, product_titles
        ):
            totals[cosine_similarity] += dot(query, product_title) / (norm(query) * norm(product_title))
            counts[cosine_similarity] += 1
        for label in labels:
            print(f"{label}: {totals[label] / counts[label]}")

    if "baseline" == args.subcommand:
        model = get_sentence_transformer("line-corporation/line-distilbert-base-japanese")
        test(model)
        # For example,
        # 0.0: 0.9513349905107426
        # 1.0: 0.959761315959156

    if "fine-tune" == args.subcommand:  # takes 10 minutes with an RTX 4080
        # Ref: https://www.sbert.net/docs/training/overview.html
        from sentence_transformers import InputExample
        from sentence_transformers.losses import CosineSimilarityLoss
        from torch.utils.data import DataLoader

        from pandas import read_csv

        merged_jp_train = read_csv(args.train_file, sep="\t")

        train_examples = []
        for cosine_similarity, query, product_title in zip(
            merged_jp_train["cosine_similarity"], merged_jp_train["query"], merged_jp_train["product_title"]
        ):
            train_examples.append(InputExample(label=cosine_similarity, texts=[query, product_title]))

        model = get_sentence_transformer("line-corporation/line-distilbert-base-japanese")
        model.fit(
            train_objectives=[
                (
                    DataLoader(train_examples, shuffle=True, batch_size=args.batch_size),
                    CosineSimilarityLoss(model),
                )
            ],
            epochs=1,
            warmup_steps=100,
            output_path=args.model_dir,
        )

    if "test" == args.subcommand:
        # Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html
        model = get_sentence_transformer(args.model_dir)
        test(model)
        # For example,
        # 0.0: 0.6788792977866293
        # 1.0: 0.8989484219986726
