import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


import json
from torch.utils.data.dataloader import default_collate
from datasets.ner.bio import convert_logits_to_examples, convert_examples_to_feature
from datasets.ner import EntityExample, NerExample
from transformers import BertTokenizerFast
import pytorch_lightning as pl
from pl_models.ner import Ner
from metrics.ner import calc_metric
from dataclasses import asdict


pl.seed_everything(2)


# load label
ID_TO_LABEL_MAP = {}
LABEL_TO_ID_MAP = {}
with open('/home/zz/data/datasets/201/labels.txt', encoding='utf-8')as f:
    index = 0
    for line in f:
        label = line.strip()
        if label:
            ID_TO_LABEL_MAP[index] = label
            LABEL_TO_ID_MAP[label] = index
            index += 1

ID_TO_LABEL_MAP[len(ID_TO_LABEL_MAP)] = 'crf'
LABEL_TO_ID_MAP['crf'] = len(LABEL_TO_ID_MAP)
num_labels = len(ID_TO_LABEL_MAP)
max_seq_length=128
BATCH_SIZE = 64
PRETRAIN_MODEL_PATH='/home/zz/data/models/albert/albert_chinese_tiny'
PAD_ID = LABEL_TO_ID_MAP['crf']



# load dataset
with open('/home/zz/data/datasets/201/201.json', encoding='utf-8') as f:
    example_list = [
        NerExample(
            text=data['MesgType_DocumentsInformation_DocumentsDetails'],
            words=list(data['MesgType_DocumentsInformation_DocumentsDetails']),
            entities=[
                EntityExample(
                    start_idx=entity['start'],
                    end_idx=entity['end'],
                    entity=entity['text'],
                    type=entity['labels'][0]
                )
                for entity in data.get('label', [])
            ]
        )
        for data in json.load(f)
    ]


# with open('diff.json', encoding='utf-8') as f:
#     example_list = [
#         NerExample(
#             text=data['text'],
#             words=list(data['text']),
#             entities=[
#                 EntityExample(
#                     start_idx=entity['start_idx'],
#                     end_idx=entity['end_idx'],
#                     entity=entity['entity'],
#                     type=entity['type']
#                 )
#                 for entity in data['entities']
#             ]
#         )
#         for data in json.load(f)
#     ]


tokenizer = BertTokenizerFast.from_pretrained(PRETRAIN_MODEL_PATH)
tokenizer.add_special_tokens({'additional_special_tokens': [' ', '\n']})


model = Ner.load_from_checkpoint('/home/zz/data/outputs/zhang_zhao/201_token_classification/ckpt/epoch=19-val_loss=7.44-v1.ckpt')
model.eval()
feature_list = convert_examples_to_feature(tokenizer, example_list, LABEL_TO_ID_MAP, max_seq_length, pad_id=PAD_ID)
batch = default_collate([
    {
        'input_ids': feature.input_ids,
        'attention_mask': feature.attention_mask,
        'token_type_ids': feature.token_type_ids,
    }
    for feature in feature_list
])

logits, *_ = model(**batch)
predict_example_list = convert_logits_to_examples(feature_list, logits, ID_TO_LABEL_MAP, pad_id=PAD_ID)



metric = calc_metric(example_list, predict_example_list)
print(metric)

with open('diff.json', 'w', encoding='utf8')as f:
    diff_list = []
    for gold_example, pred_example in zip(example_list, predict_example_list):
        if gold_example.entities != pred_example.entities:
            diff_list.append({
                'text': gold_example.text,
                # 'words': gold_example.words,
                'entities': [asdict(entity_example) for entity_example in gold_example.entities],
                'preds': [asdict(entity_example) for entity_example in pred_example.entities],
            })
    json.dump(diff_list, f, indent=2, ensure_ascii=False)

