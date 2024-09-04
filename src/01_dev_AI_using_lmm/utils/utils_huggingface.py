import typing


def inference_results(query_list: typing.List, infer_results: typing.List) -> bool:
    if len(query_list) == len(infer_results) == 0:
        return False
    if len(query_list) != len(infer_results):
        return False

    for idx, (q, res) in enumerate(zip(query_list, infer_results)):
        print(f"[{idx}] Query: {q}\n    ===> label: {res['label']}, score: {res['score']}")
    print()

    return True
