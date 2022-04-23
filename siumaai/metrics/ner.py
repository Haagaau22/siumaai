from typing import List
from siumaai.features.ner import NerExample


def calc_metric(gold_example_list: List[NerExample], pred_example_list: List[NerExample]):
    tp, fp, fn = 0, 0, 0
    for gold_example, pred_example in zip(gold_example_list, pred_example_list):
        _tp, _fp, _fn = 0, 0, 0
        for dst_entity in pred_example.entities:
            if dst_entity in gold_example.entities:
                _tp += 1
            else:
                _fp += 1
        _fn = len(gold_example.entities) - _tp
        tp += _tp
        fp += _fp
        fn += _fn
    precision = tp / (tp+fp) if tp+fp != 0 else 0
    recall = tp / (tp+fn) if tp+fn != 0 else 0
    f1 = 2 * precision * recall / (precision+recall) if precision+recall != 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
