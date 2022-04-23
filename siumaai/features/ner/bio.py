import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from . import NerExample, NerFeature, EntityExample
from ..utils import add_pad_for_labels, remove_pad_for_labels, compare_words_tokens


@dataclass
class BIOForNerFeature(NerFeature):
    labels: Optional[torch.Tensor]=None


def convert_examples_to_feature(
        tokenizer,
        example_list: List[NerExample],
        label_to_id_map: Dict,
        max_seq_length: int,
        pad_id: int=-100,
        label_all_tokens: bool=False,
        check_tokenization=False
        ) -> List[BIOForNerFeature]:

    
    encoded_inputs = tokenizer(
        [example.words for example in example_list],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=max_seq_length
    )

    feature_list = []
    for index, example in enumerate(example_list):
        word_ids = encoded_inputs.word_ids(index)
        assert check_tokenization is False or compare_words_tokens(tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][index]), example.words, word_ids), example

        len_words = len(example.words)
        labels = [label_to_id_map['O'] for _ in range(len_words)]

        # read labels
        for entity in example.entities:
            if len_words < entity.end_idx or ''.join(example.words[entity.start_idx: entity.end_idx]) != entity.entity:
                guess_start = example.text.index(entity.entity)
                guess_end = guess_start + len(entity.entity)
                raise Exception(f'index error, text: {example.text}, item: {entity}, max_len: {len_words}, guess: {(guess_start, guess_end)}')

            labels[entity.start_idx] = label_to_id_map[f'B-{entity.type}']
            for entity_index in range(entity.start_idx+1, entity.end_idx):
                if entity_index < len_words:
                    labels[entity_index] = label_to_id_map[f'I-{entity.type}']

        # print(f'word_ids: {word_ids)}')
        # print(tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][index]))

        # fix labels
        labels = add_pad_for_labels(word_ids, labels, pad_id, encoded_inputs['token_type_ids'][index], label_all_tokens)


        feature_list.append(BIOForNerFeature(
            text=example.text,
            entities=example.entities,
            words=example.words,
            word_ids=word_ids,
            input_ids=torch.tensor(encoded_inputs['input_ids'][index]),
            attention_mask=torch.tensor(encoded_inputs['attention_mask'][index]),
            token_type_ids=torch.tensor(encoded_inputs['token_type_ids'][index]),
            labels=torch.tensor(labels)
        ))

    return feature_list



class BIOForNerDataset(Dataset):
    def __init__(self, example_list: List[NerExample], tokenizer, label_to_id_map, max_seq_length, pad_id=-100, label_all_tokens=False, check_tokenization=False, lazy_load=False):
        self.example_list = example_list
        self.tokenizer = tokenizer
        self.label_to_id_map = label_to_id_map
        self.max_seq_length = max_seq_length
        self.pad_id = pad_id
        self.label_all_tokens = label_all_tokens
        self.check_tokenization = check_tokenization
        self.lazy_load = lazy_load
        

        if lazy_load is False:
            self.feature_list = convert_examples_to_feature(
                    self.tokenizer, 
                    self.example_list, 
                    self.label_to_id_map, 
                    self.max_seq_length, 
                    self.pad_id,
                    label_all_tokens=self.label_all_tokens,
                    check_tokenization=self.check_tokenization)

    def __len__(self):
        # return len(self.feature_list)
        return len(self.example_list)

    def __getitem__(self, idx) -> BIOForNerFeature:
        if self.lazy_load is False:
            return self.feature_list[idx]
        else:
            return convert_examples_to_feature(
                    self.tokenizer, 
                    [self.example_list[idx]], 
                    self.label_to_id_map, 
                    self.max_seq_length, 
                    self.pad_id,
                    label_all_tokens=self.label_all_tokens,
                    check_tokenization=self.check_tokenization)[0]

def convert_logits_to_examples(feature_list: List[BIOForNerFeature], logits: torch.Tensor, id_to_label_map: Dict) -> List[NerExample]:

    example_list = []
    for logit, feature in zip(logits, feature_list):
        label_id_list = logit.argmax(axis=1).tolist()
        label_id_list = remove_pad_for_labels(feature.word_ids, label_id_list, feature.token_type_ids)
        label_list = [id_to_label_map[label_id] for label_id in label_id_list]

        entities = []
        previous_start_idx = None
        previous_name = None
        for index, label in enumerate(label_list):
            flag, *name = label.split('-')
            name = '-'.join(name)
            if previous_name is not None and previous_start_idx is not None and (flag in {'O', 'B'} or name != previous_name):
                entities.append(EntityExample(
                    start_idx=previous_start_idx,
                    end_idx=index,
                    entity=''.join(feature.words[previous_start_idx: index]),
                    type=previous_name))
                previous_start_idx = None
                previous_name = None

            if flag == 'B':
                previous_start_idx = index
                previous_name = name

        if previous_name is not None and previous_start_idx is not None:
            entities.append(EntityExample(
                start_idx=previous_start_idx,
                end_idx=len(label_list),
                entity=''.join(feature.words[previous_start_idx: len(label_list)]),
                type=previous_name))

        example_list.append(NerExample(text=feature.text, entities=entities, words=feature.words))
    return example_list


def convert_crf_logits_to_examples(feature_list: List[BIOForNerFeature], logits: torch.Tensor, id_to_label_map: Dict) -> List[NerExample]:

    example_list = []
    for logit, feature in zip(logits[0], feature_list):

        label_id_list = remove_pad_for_labels(feature.word_ids, logit.tolist(), feature.token_type_ids)
        label_list = [id_to_label_map[label_id] for label_id in label_id_list]

        entities = []
        previous_start_idx = None
        previous_name = None
        for index, label in enumerate(label_list):
            flag, *name = label.split('-')
            name = '-'.join(name)
            if previous_name is not None and previous_start_idx is not None and (flag in {'O', 'B'} or name != previous_name):
                entities.append(EntityExample(
                    start_idx=previous_start_idx,
                    end_idx=index,
                    entity=''.join(feature.words[previous_start_idx: index]),
                    type=previous_name))
                previous_start_idx = None
                previous_name = None

            if flag == 'B':
                previous_start_idx = index
                previous_name = name

        if previous_name is not None and previous_start_idx is not None:
            entities.append(EntityExample(
                start_idx=previous_start_idx,
                end_idx=len(label_list),
                entity=''.join(feature.words[previous_start_idx: len(label_list)]),
                type=previous_name))

        example_list.append(NerExample(text=feature.text, entities=entities, words=feature.words))
    return example_list
