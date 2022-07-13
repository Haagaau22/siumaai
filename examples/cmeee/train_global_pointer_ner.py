import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


import json 
from dataclasses import asdict
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataloader import default_collate
from siumaai.features.ner.global_pointer import GlobalPointerForNerDataset, convert_logits_to_examples
from siumaai.features.ner import EntityExample, NerExample
from transformers import BertTokenizerFast
import pytorch_lightning as pl
from siumaai.pl_models.ner import Ner
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoConfig
from siumaai.models import MODEL_CLS_MAP


pl.seed_everything(2)


# load label
# LABEL_TO_ID_MAP = {
#     'ns': 0, 
#     'nt': 1,
#     'nr':2,
# }
# ID_TO_LABEL_MAP = {
#     0: 'ns',
#     1: 'nt',
#     2: 'nr'
# }

LABEL_TO_ID_MAP = {}
ID_TO_LABEL_MAP = {}
with open('/content/drive/MyDrive/datasets/CMeEE/labels.txt', encoding='utf8')as f:
    index = 0
    for line in f:
        label = line.strip()
        if label:
            LABEL_TO_ID_MAP[label] = index
            ID_TO_LABEL_MAP[index] = label
            index += 1

NUM_LABELS = len(LABEL_TO_ID_MAP)
MAX_SEQ_LENGTH= 128
BATCH_SIZE = 120
PRETRAIN_MODEL_PATH='bert-base-chinese'
PAD_ID = -100



# load dataset
with open('/content/drive/MyDrive/datasets/CMeEE/data.json', encoding='utf-8') as f:
    example_list = [
        NerExample(
            text=data['text'],
            words=list(data['text']),
            entities=[
                EntityExample(
                    start_idx=entity['start_idx'],
                    end_idx=entity['end_idx'],
                    entity=entity['entity'],
                    type=entity['type']
                )
                for entity in data.get('entities', [])
            ]
        )
        for data in json.load(f)
    ]


tokenizer = BertTokenizerFast.from_pretrained(PRETRAIN_MODEL_PATH)
tokenizer.add_special_tokens({'additional_special_tokens': [' ', '\n']})

train_example_size = int(len(example_list) * 0.8)
val_example_size = int(len(example_list) * 0.1)
test_example_size = len(example_list) - train_example_size - val_example_size
train_example_list, val_example_list, test_example_list = random_split(
        example_list, [train_example_size, val_example_size, test_example_size])


train_dataset = GlobalPointerForNerDataset(train_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False, lazy_load=True)
val_dataset = GlobalPointerForNerDataset(val_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False, lazy_load=True)

print(f'train_dataset_size: {len(train_dataset)}')
print(f'val_dataset_size: {len(val_dataset)}')


def fit_collate_func(batch):
    return default_collate([
        {
            'input_ids': data.input_ids,
            'attention_mask': data.attention_mask,
            'token_type_ids': data.token_type_ids,
            'labels': data.labels,
            'criterion_mask': data.criterion_mask,
        }
        for data in batch
    ])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fit_collate_func)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fit_collate_func)





if len(sys.argv) == 1 or sys.argv[1] == 'train':
    
    config = AutoConfig.from_pretrained(
        PRETRAIN_MODEL_PATH, 
        return_dict=None)

    model_cls = MODEL_CLS_MAP['global_pointer_for_ner']
    model_kwargs = {
        'pretrain_model_path': PRETRAIN_MODEL_PATH,
        'inner_dim': 64,
        'hidden_size': config.hidden_size,
        'num_labels': NUM_LABELS,
        'vocab_len': len(tokenizer)
    }

    model = Ner(
            learning_rate=0.0001737800828,
            adam_epsilon=1e-8,
            warmup_rate=0.1,
            weight_decay=0.1,
            model_cls=model_cls,
            **model_kwargs
            )

    trainer = Trainer(
            gpus=1,
            # max_epochs=20,
            max_epochs=10,
            weights_summary=None,
            logger=TensorBoardLogger('tensorboard_logs/global_pointer'),
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.005,
                    patience=2,
                    verbose=False,
                    mode='min'),
                ModelCheckpoint(
                    dirpath='ckpt/global_pointer',
                    filename='{epoch}-{val_loss:.2f}',
                    monitor='val_loss',
                    mode='min',
                    verbose=True,
                    save_top_k=1),
                LearningRateMonitor(logging_interval='step')])

    # lr = trainer.tuner.lr_find(model, train_dataloader, val_dataloader, early_stop_threshold=None)
    # print(lr.suggestion())
    # model.hparams.learning_rate = lr.suggestion()

    trainer.fit(model, train_dataloader, val_dataloader)

elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    import torch
    from siumaai.metrics.ner import calc_metric
    TEST_BATCH_SIZE = 80
    model = Ner.load_from_checkpoint('ckpt/global_pointer/epoch=2-val_loss=12.92.ckpt')
    model.to('cuda')
    model.eval()


    test_dataset = GlobalPointerForNerDataset(test_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False)

    pred_example_list = []
    start_index = 0
    while start_index < len(test_dataset):
        if start_index + TEST_BATCH_SIZE < len(test_dataset):
            end_index  = start_index + TEST_BATCH_SIZE 
        else:
            end_index = len(test_dataset)

        feature_list = []
        batch = []
        for index in range(start_index, end_index):
            feature_list.append(test_dataset[index])
            batch.append({
                'input_ids': test_dataset[index].input_ids,
                'attention_mask': test_dataset[index].attention_mask,
                'token_type_ids': test_dataset[index].token_type_ids,
            })

        with torch.no_grad():
            batch = {k:v.to('cuda') for k,v in default_collate(batch).items()}
            *_, logits = model(**batch)
            logits = logits.detach().cpu()
        pred_example_list.extend(convert_logits_to_examples(feature_list, logits, ID_TO_LABEL_MAP))
        print(f'finish {start_index} -> {end_index}')
        start_index = end_index
    print(f'test_example_list: {len(test_example_list)}, pred_example_list: {len(pred_example_list)}')


    metric = calc_metric(test_example_list, pred_example_list)
    print(metric)

    if not os.path.isdir('pred/global_pointer'):
        os.makedirs('pred/global_pointer')

    with open('pred/global_pointer/diff.json', 'w', encoding='utf8')as f:
        diff_list = []
        for gold_example, pred_example in zip(test_example_list, pred_example_list):
            if gold_example.entities != pred_example.entities:
                diff_list.append({
                    'text': gold_example.text,
                    'entities': [asdict(entity_example) for entity_example in gold_example.entities],
                    'preds': [asdict(entity_example) for entity_example in pred_example.entities],
                })
        json.dump(diff_list, f, indent=2, ensure_ascii=False)
