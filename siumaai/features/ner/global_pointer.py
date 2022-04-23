import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from . import NerExample, NerFeature, EntityExample
from ..utils import add_pad_for_3d_labels, remove_pad_for_3d_labels, compare_words_tokens


@dataclass
class GlobalPointerForNerFeature(NerFeature):
    labels: torch.Tensor
    criterion_mask: torch.Tensor


def convert_examples_to_feature(
        tokenizer,
        example_list: List[NerExample],
        label_to_id_map: Dict,
        max_seq_length: int,
        pad_id: int=-100,
        label_all_tokens: bool=False,
        check_tokenization=False
        ) -> List[GlobalPointerForNerFeature]:

    
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

        labels = [
            [
                [ 0 for _ in example.words]
                for _ in example.words
            ]
            for _ in label_to_id_map
        ]

        # read labels
        for entity in example.entities:
            if len_words < entity.end_idx or ''.join(example.words[entity.start_idx: entity.end_idx]) != entity.entity:
                guess_start = example.text.index(entity.entity)
                guess_end = guess_start + len(entity.entity)
                raise Exception(f'index error, text: {example.text}, item: {entity}, max_len: {len_words}, guess: {(guess_start, guess_end)}')
            labels[label_to_id_map[entity.type]][entity.start_idx][entity.end_idx-1] = 1


        # fix labels
        labels, criterion_mask = add_pad_for_3d_labels(
                word_ids, 
                labels, 
                pad_id, 
                encoded_inputs['token_type_ids'][index], 
                label_all_tokens, 
                return_criterion_mask=True)


        feature_list.append(GlobalPointerForNerFeature(
            text=example.text,
            entities=example.entities,
            words=example.words,
            word_ids=word_ids,
            input_ids=torch.tensor(encoded_inputs['input_ids'][index]),
            attention_mask=torch.tensor(encoded_inputs['attention_mask'][index]),
            token_type_ids=torch.tensor(encoded_inputs['token_type_ids'][index]),
            labels=torch.tensor(labels),
            criterion_mask=torch.tensor(criterion_mask)
        ))

    return feature_list



class GlobalPointerForNerDataset(Dataset):
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

    def __getitem__(self, idx) -> GlobalPointerForNerFeature:
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


def convert_logits_to_examples(
        feature_list: List[GlobalPointerForNerFeature], 
        logits: torch.Tensor, 
        id_to_label_map: Dict, 
        remove_overlap=None) -> List[NerExample]:
    """
    remove_overlap去重， 可取值:
        None: 不去重
        'short': 去除短的
        'long': 去除长的
    """

    example_list = []

    for logit, feature in zip(logits, feature_list):

        labels = torch.where(logit>0, 1, 0).tolist()
        labels = remove_pad_for_3d_labels(feature.word_ids, labels, feature.token_type_ids)

        entities = []
        for label_id, start_end_labels in enumerate(labels):
            previous_end_index = None
            for start_index, end_labels in enumerate(start_end_labels):
                for end_index, span_label in enumerate(end_labels):

                    if end_index >= start_index and span_label == 1:
                        if  remove_overlap == 'short' and end_index == previous_end_index:
                            continue
                        elif  remove_overlap == 'long' and end_index == previous_end_index:
                            entities.pop()

                        entities.append(EntityExample(
                            start_idx=start_index,
                            end_idx=end_index+1,
                            entity=''.join(feature.words[start_index: end_index+1]),
                            type=id_to_label_map[label_id]))
                        previous_end_index = end_index
                        break
        
        example_list.append(NerExample(
            text=feature.text, 
            entities=sorted(entities, key=lambda entity: entity.start_idx), 
            words=feature.words))

    return example_list
