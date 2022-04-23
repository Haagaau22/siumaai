import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


import json
from dataclasses import asdict
from torch.utils.data import random_split, DataLoader
from siumaai.features.ner.bio import convert_logits_to_examples, convert_crf_logits_to_examples
from torch.utils.data.dataloader import default_collate
from siumaai.features.ner.bio import BIOForNerDataset
from siumaai.features.ner import EntityExample, NerExample
from transformers import BertTokenizerFast
import pytorch_lightning as pl
from siumaai.pl_models.ner import CrfNer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoConfig
from siumaai.models import MODEL_CLS_MAP


pl.seed_everything(2)


# load label
ID_TO_LABEL_MAP = {}
LABEL_TO_ID_MAP = {}
with open('msra/ner/labels.txt', encoding='utf-8')as f:
    index = 0
    for line in f:
        label = line.strip()
        if label and label != 'O':
            ID_TO_LABEL_MAP[index] = f'B-{label}'
            LABEL_TO_ID_MAP[f'B-{label}'] = index
            index += 1

            ID_TO_LABEL_MAP[index] = f'I-{label}'
            LABEL_TO_ID_MAP[f'I-{label}'] = index
            index += 1

        elif label and label == 'O':
            ID_TO_LABEL_MAP[index] = label
            LABEL_TO_ID_MAP[label] = index
            index += 1

ID_TO_LABEL_MAP[len(ID_TO_LABEL_MAP)] = '[PAD]'
LABEL_TO_ID_MAP['[PAD]'] = len(LABEL_TO_ID_MAP)

NUM_LABELS = len(ID_TO_LABEL_MAP)
MAX_SEQ_LENGTH=128
BATCH_SIZE = 240
# PRETRAIN_MODEL_PATH='/home/zz/data/models/albert/albert_chinese_tiny'
PRETRAIN_MODEL_PATH='clue/albert_chinese_tiny'
PAD_ID = LABEL_TO_ID_MAP['[PAD]']
# PAD_ID = -100

print(f'id_to_label_map: {ID_TO_LABEL_MAP}')
print(f'label_to_id_map: {LABEL_TO_ID_MAP}')


# load examples
with open('msra/ner/data.json', encoding='utf-8') as f:
    example_list = [
        NerExample(
            text=data['text'],
            words=list(data['text']),
            entities=[
                EntityExample(
                    start_idx=entity['start_idx'],
                    end_idx=entity['end_idx'],
                    entity=entity['entity'],
                    type=entity['label']
                )
                for entity in data.get('entities', [])
            ]
        )
        for data in json.load(f)
    ]

train_example_size = int(len(example_list) * 0.8)
val_example_size = int(len(example_list) * 0.1)
test_example_size = len(example_list) - train_example_size - val_example_size
train_example_list, val_example_list, test_example_list = random_split(
        example_list, [train_example_size, val_example_size, test_example_size])


### load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRETRAIN_MODEL_PATH)
tokenizer.add_special_tokens({'additional_special_tokens': [' ', '\n']})





if len(sys.argv) == 1 or sys.argv[1] == 'train':

    ### load train/val dataloader
    train_dataset = BIOForNerDataset(train_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False)
    val_dataset = BIOForNerDataset(val_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False)

    print(f'train_dataset_size: {len(train_dataset)}')
    print(f'val_dataset_size: {len(val_dataset)}')


    def fit_collate_func(batch):
        return default_collate([
            {
                'input_ids': data.input_ids,
                'attention_mask': data.attention_mask,
                'token_type_ids': data.token_type_ids,
                'labels': data.labels
            }
            for data in batch
        ])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fit_collate_func)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fit_collate_func)

    ### init model
    config = AutoConfig.from_pretrained(
        PRETRAIN_MODEL_PATH, 
        return_dict=None)

    model_cls = MODEL_CLS_MAP['crf_for_ner']
    model_kwargs = {
        'pretrain_model_path': PRETRAIN_MODEL_PATH,
        'num_labels': NUM_LABELS,
        'dropout_rate': config.hidden_dropout_prob,
        'hidden_size': config.hidden_size,
        'vocab_len': len(tokenizer)
    }
    # model = Ner(
    model = CrfNer(
            # crf_learning_rate=0.005248074602497723,
            # learning_rate=0.0005248074602497723,
            crf_learning_rate=0.001,
            learning_rate=0.0001,
            adam_epsilon=1e-8,
            warmup_rate=0.1,
            weight_decay=0.1,
            model_cls=model_cls,
            **model_kwargs
            )


    ### init trainer
    trainer = Trainer(
            gpus=1,
            max_epochs=100,
            weights_summary=None,
            logger=TensorBoardLogger('tensorboard_logs'),
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.1,
                    patience=5,
                    verbose=False,
                    mode='min'),
                ModelCheckpoint(
                    dirpath='ckpt',
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
    from siumaai.metrics.ner import calc_metric
    TEST_BATCH_SIZE = 8

    ### load test dataset
    test_dataset = BIOForNerDataset(test_example_list, tokenizer, LABEL_TO_ID_MAP, MAX_SEQ_LENGTH, pad_id=PAD_ID, check_tokenization=False)

    ### load model
    # model = Ner.load_from_checkpoint('ckpt/epoch=19-val_loss=7.44-v10.ckpt')
    model = CrfNer.load_from_checkpoint('ckpt/epoch=9-val_loss=2.09.ckpt')
    model.eval()
    print(model.model.crf.transitions)
    print(LABEL_TO_ID_MAP)


    pred_example_list = []
    crf_pred_example_list = []
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

        crf_logits, logits, *_ = model(**default_collate(batch))
        pred_example_list.extend(convert_logits_to_examples(feature_list, logits, ID_TO_LABEL_MAP))
        crf_pred_example_list.extend(convert_crf_logits_to_examples(feature_list, crf_logits, ID_TO_LABEL_MAP))
        print(f'finish {start_index} -> {end_index}')
        start_index = end_index


    metric = calc_metric(test_example_list, pred_example_list)
    print(metric)

    crf_metric = calc_metric(test_example_list, crf_pred_example_list)
    print(crf_metric)

    if not os.path.isdir('pred'):
        os.mkdir('pred')

    with open('pred/diff.json', 'w', encoding='utf8')as f:
        diff_list = []
        for gold_example, pred_example in zip(test_example_list, pred_example_list):
            if gold_example.entities != pred_example.entities:
                diff_list.append({
                    'text': gold_example.text,
                    'entities': [asdict(entity_example) for entity_example in gold_example.entities],
                    'preds': [asdict(entity_example) for entity_example in pred_example.entities],
                })
        json.dump(diff_list, f, indent=2, ensure_ascii=False)
