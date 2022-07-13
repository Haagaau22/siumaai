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


def calc_metric2(gold_example_list: List[NerExample], pred_example_list: List[NerExample]):
    tp_map, fp_map, fn_map = {}, {}, {}

    for gold_example, pred_example in zip(gold_example_list, pred_example_list):
        for pred_entity in pred_example.entities:
            if pred_entity in gold_example.entities:
                if pred_entity.type not in tp_map:
                    tp_map[pred_entity.type] = 0
                tp_map[pred_entity.type] += 1
            else:
                if pred_entity.type not in fp_map:
                    fp_map[pred_entity.type] = 0
                fp_map[pred_entity.type] += 1

        for gold_entity in gold_example.entities:
            if gold_entity not in pred_example.entities:
                if gold_entity.type not in fn_map:
                    fn_map[gold_entity.type] = 0
                fn_map[gold_entity.type] += 1

    label_set = set()
    label_set.update(tp_map.keys())
    label_set.update(fp_map.keys())
    label_set.update(fn_map.keys())

    metric = {
        label: {
            'precision': tp_map[label] / (tp_map[label]+fp_map[label]) if tp_map[label]+fp_map[label] != 0 else 0,
            'recall': tp_map[label] / (tp_map[label]+fn_map[label]) if tp_map[label]+fn_map[label] != 0 else 0,
            'f1': 2 * tp_map[label] / (2 * tp_map[label] + fn_map[label] + fp_map[label]) if 2 * tp_map[label] + fn_map[label] + fp_map[label] != 0 else 0
        }
        for label in label_set
    }

    metric['micro'] = {
        'precision': sum(tp_map.values()) / (sum(tp_map.values()) + sum(fp_map.values())) if sum(tp_map.values())+sum(fp_map.values()) != 0 else 0,
        'recall': sum(tp_map.values()) / (sum(tp_map.values()) + sum(fn_map.values())) if sum(tp_map.values())+sum(fn_map.values()) != 0 else 0,
        'f1': 2 * sum(tp_map.values()) / (2 * sum(tp_map.values()) + sum(fn_map.values()) + sum(fp_map.values())) if 2 * sum(tp_map.values()) + sum(fn_map.values()) + sum(fp_map.values()) != 0 else 0
    }

    metric['macro'] = {
        item: sum([metric[label][item] for label in label_set]) / len(label_set)
        for item in ['precision', 'recall', 'f1']
    }

    return metric
