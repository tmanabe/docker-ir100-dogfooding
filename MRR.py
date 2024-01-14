#!/usr/bin/env python


def calc_mrr(
    data_frame, query_column="query", actual_column="actual", document_column="document", expect_column="expect"
):
    data_frame.sort_values([query_column, actual_column, document_column], ascending=[True, False, True], inplace=True)
    last_query, rr, total, count, rank = None, None, 0.0, 0, 0
    for query, expect in zip(data_frame[query_column], data_frame[expect_column]):
        if last_query != query:
            if rr is not None:
                total += rr
                count += 1
            last_query, rr, rank = query, None, 0
        rank += 1
        if rr is None and 0 < float(expect):
            rr = 1.0 / rank
    if rr is not None:
        total += rr
        count += 1
    if 0 < count:
        return round(total / count, 3)
    else:
        return None


if "__main__" == __name__:
    from pandas import DataFrame

    subject = DataFrame(
        {
            "query": [],
            "actual": [],
            "expect": [],
        }
    )
    assert calc_mrr(subject) is None

    subject = DataFrame(
        {
            "query": [1, 1, 2, 2, 2, 0, 0, 0, 0],
            "actual": [5, 2, 6, 8, 1, 0, 4, 3, 9],
            "expect": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    assert round((1 / 2 + 1 / 1 + 1 / 3) / 3, 3) == round(calc_mrr(subject), 3)

    subject = subject.sample(frac=1, random_state=0)
    assert round((1 / 2 + 1 / 1 + 1 / 3) / 3, 3) == round(calc_mrr(subject), 3)
