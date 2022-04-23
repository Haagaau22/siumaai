import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from . import NerExample, NerFeature, EntityExample
from ..utils import add_pad_for_labels, remove_pad_for_labels, compare_words_tokens


@dataclass
class SpanForNerFeature(NerFeature):
    start_labels: Optional[torch.Tensor]=None
    end_labels: Optional[torch.Tensor]=None


def convert_examples_to_feature(
        tokenizer,
        example_list: List[NerExample],
        label_to_id_map: Dict,
        max_seq_length: int,
        pad_id: int=-100,
        label_all_tokens: bool=False,
        check_tokenization=False
        ) -> List[SpanForNerFeature]:

    
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
        start_labels = [label_to_id_map['O'] for _ in range(len_words)]
        end_labels = [label_to_id_map['O'] for _ in range(len_words)]

        # read labels
        for entity in example.entities:
            if len_words < entity.end_idx or ''.join(example.words[entity.start_idx: entity.end_idx]) != entity.entity:
                guess_start = example.text.index(entity.entity)
                guess_end = guess_start + len(entity.entity)
                raise Exception(f'index error, text: {example.text}, item: {entity}, max_len: {len_words}, guess: {(guess_start, guess_end)}')

            start_labels[entity.start_idx] = label_to_id_map[entity.type]
            end_labels[entity.end_idx-1] = label_to_id_map[entity.type]

        # fix labels
        start_labels = add_pad_for_labels(word_ids, start_labels, pad_id, encoded_inputs['token_type_ids'][index], label_all_tokens)
        end_labels = add_pad_for_labels(word_ids, end_labels, pad_id, encoded_inputs['token_type_ids'][index], label_all_tokens)


        feature_list.append(SpanForNerFeature(
            text=example.text,
            entities=example.entities,
            words=example.words,
            word_ids=word_ids,
            input_ids=torch.tensor(encoded_inputs['input_ids'][index]),
            attention_mask=torch.tensor(encoded_inputs['attention_mask'][index]),
            token_type_ids=torch.tensor(encoded_inputs['token_type_ids'][index]),
            start_labels=torch.tensor(start_labels),
            end_labels=torch.tensor(end_labels)
        ))

    return feature_list



class SpanForNerDataset(Dataset):
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

    def __getitem__(self, idx) -> SpanForNerFeature:
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


def convert_logits_to_examples(feature_list: List[SpanForNerFeature], start_logits: torch.Tensor, end_logits: torch.Tensor, id_to_label_map: Dict, remove_overlap=None) -> List[NerExample]:
    """
    remove_overlap去重， 可取值:
        None: 不去重
        'short': 去除短的
        'long': 去除长的
    """

    example_list = []
    for start_logit, end_logit, feature in zip(start_logits, end_logits, feature_list):
        start_label_id_list = start_logit.argmax(axis=1).tolist()
        start_label_id_list = remove_pad_for_labels(feature.word_ids, start_label_id_list, feature.token_type_ids)
        start_label_list = [id_to_label_map[label_id] for label_id in start_label_id_list]

        end_label_id_list = end_logit.argmax(axis=1).tolist()
        end_label_id_list = remove_pad_for_labels(feature.word_ids, end_label_id_list, feature.token_type_ids)
        end_label_list = [id_to_label_map[label_id] for label_id in end_label_id_list]

        entities = []

        previous_end_idx = None
        for start_idx, start_label in enumerate(start_label_list):
            if start_label == 'O':
                continue

            for end_idx, end_label in enumerate(end_label_list):
                if end_idx >= start_idx and start_label == end_label:
                    if  remove_overlap == 'short' and end_idx == previous_end_idx:
                        continue
                    elif  remove_overlap == 'long' and end_idx == previous_end_idx:
                        entities.pop()

                    entities.append(EntityExample(
                        start_idx=start_idx,
                        end_idx=end_idx+1,
                        entity=''.join(feature.words[start_idx: end_idx+1]),
                        type=start_label))
                    previous_end_idx = end_idx
                    break

        example_list.append(NerExample(text=feature.text, entities=entities, words=feature.words))
    return example_list
