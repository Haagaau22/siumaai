import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Any, Union
from . import NerExample, NerFeature, EntityExample
from ..utils import add_pad_for_labels, remove_pad_for_labels, compare_words_tokens, add_pad_for_2d_labels, remove_pad_for_2d_labels


@dataclass
class MRCForNerExample:
    query_words: List
    context_words: List
    text: str
    query_type: Any
    start_labels: List
    end_labels: List
    span_labels: List[List]
    example_id: int


@dataclass
class MRCForNerFeature:
    query_type: Any
    example_id: int
    query_words: List[str]
    context_words: List[str]
    word_ids: List[Union[int, None]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    start_labels: torch.Tensor
    end_labels: torch.Tensor
    span_labels: torch.Tensor
    start_criterion_mask: torch.Tensor
    end_criterion_mask: torch.Tensor
    span_criterion_mask: torch.Tensor



class MRCForNerDataset(Dataset):
    def __init__(self, 
            example_list: List[NerExample], 
            tokenizer, 
            label_to_query_map,
            max_seq_length, 
            pad_id=-100, 
            label_all_tokens=False, 
            check_tokenization=False,
            lazy_load=False):
        self.example_list = example_list
        self.tokenizer = tokenizer
        self.label_to_query_map = label_to_query_map
        self.max_seq_length = max_seq_length
        self.pad_id = pad_id
        self.label_all_tokens = label_all_tokens
        self.check_tokenization = check_tokenization
        self.lazy_load = lazy_load

        self.mrc_example_list = []
        for example_id, example in enumerate(self.example_list):
            for query_type, query in label_to_query_map.items():
            
                start_labels = [0. for _ in example.words]
                end_labels = [0. for _ in example.words]
                span_labels = [[0. for _ in example.words] for _ in example.words]

                for entity in example.entities:
                    if entity.type != query_type:
                        continue
                    if len(example.words) < entity.end_idx or ''.join(example.words[entity.start_idx: entity.end_idx]) != entity.entity:
                        guess_start = example.text.index(entity.entity)
                        guess_end = guess_start + len(entity.entity)
                        raise Exception(f'index error, text: {example.text}, item: {entity}, max_len: {len(example.words)}, guess: {(guess_start, guess_end)}')

                    start_labels[entity.start_idx] = 1.0
                    end_labels[entity.end_idx-1] = 1.0
                    span_labels[entity.start_idx][entity.end_idx-1] = 1.0
                self.mrc_example_list.append(MRCForNerExample(
                    list(query),
                    example.words,
                    example.text,
                    query_type,
                    start_labels,
                    end_labels,
                    span_labels,
                    example_id
                    ))
        

        if self.lazy_load is False:
            self.feature_list = convert_examples_to_feature(
                    self.tokenizer, 
                    self.mrc_example_list, 
                    self.max_seq_length, 
                    self.pad_id,
                    label_all_tokens=self.label_all_tokens,
                    check_tokenization=self.check_tokenization)

    def __len__(self):
        # return len(self.feature_list)
        return len(self.mrc_example_list)

    def __getitem__(self, idx) -> MRCForNerFeature:
        if self.lazy_load is False:
            return self.feature_list[idx]
        else:
            return convert_examples_to_feature(
                    self.tokenizer, 
                    [self.mrc_example_list[idx]], 
                    self.max_seq_length, 
                    self.pad_id,
                    label_all_tokens=self.label_all_tokens,
                    check_tokenization=self.check_tokenization)[0]

def convert_examples_to_feature(
        tokenizer,
        example_list: List[MRCForNerExample],
        max_seq_length: int,
        pad_id: int=-100,
        label_all_tokens: bool=False,
        check_tokenization=False
        ) -> List[MRCForNerFeature]:

    
    encoded_inputs = tokenizer(
        [example.query_words for example in example_list],
        [example.context_words for example in example_list],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=max_seq_length
    )

    feature_list = []
    for index, example in enumerate(example_list):
        word_ids = encoded_inputs.word_ids(index)
        assert check_tokenization is False or compare_words_tokens(tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][index]), example.query_words+example.context_words, word_ids), example
        token_type_ids = encoded_inputs['token_type_ids'][index]

        # fix labels
        start_labels, start_criterion_mask = add_pad_for_labels(
                word_ids, 
                example.start_labels, 
                pad_id, 
                token_type_ids, 
                label_all_tokens=label_all_tokens, 
                required_token_type_id=1,
                return_criterion_mask=True)
        end_labels, end_criterion_mask = add_pad_for_labels(
                word_ids, 
                example.end_labels, 
                pad_id, 
                token_type_ids, 
                label_all_tokens=label_all_tokens, 
                required_token_type_id=1,
                return_criterion_mask=True)
        span_labels, span_criterion_mask = add_pad_for_2d_labels(
                word_ids, 
                example.span_labels, 
                pad_id, 
                token_type_ids, 
                label_all_tokens=label_all_tokens, 
                required_token_type_id=1,
                return_criterion_mask=True)


        feature_list.append(MRCForNerFeature(
            query_type=example.query_type,
            example_id=example.example_id,
            query_words=example.query_words,
            context_words=example.context_words,
            word_ids=word_ids,
            input_ids=torch.tensor(encoded_inputs['input_ids'][index]),
            attention_mask=torch.tensor(encoded_inputs['attention_mask'][index]),
            token_type_ids=torch.tensor(encoded_inputs['token_type_ids'][index]),
            start_labels=torch.tensor(start_labels),
            end_labels=torch.tensor(end_labels),
            span_labels=torch.tensor(span_labels),
            start_criterion_mask=torch.tensor(start_criterion_mask),
            end_criterion_mask=torch.tensor(end_criterion_mask),
            span_criterion_mask=torch.tensor(span_criterion_mask)
        ))

    return feature_list


def convert_logits_to_examples(
        feature_list: List[MRCForNerFeature], 
        start_logits: torch.Tensor, 
        end_logits: torch.Tensor, 
        span_logits: torch.Tensor,
        remove_overlap=None) -> List[NerExample]:

    example_list = []
    group_example_id_map = {}
    group_example_list = []

    for index, (start_logit, end_logit, span_logit, feature) in enumerate(zip(start_logits, end_logits, span_logits, feature_list)):
        example_id = feature.example_id
        query_type = feature.query_type

        start_labels = torch.where(start_logit>0, 1, 0).tolist()
        start_labels = remove_pad_for_labels(
                feature.word_ids, start_labels, feature.token_type_ids, required_token_type_id=1)

        end_labels = torch.where(end_logit>0, 1, 0).tolist()
        end_labels = remove_pad_for_labels(
                feature.word_ids, end_labels, feature.token_type_ids, required_token_type_id=1)

        span_labels = torch.where(span_logit>0, 1, 0).tolist()
        span_labels = remove_pad_for_2d_labels(
                feature.word_ids, span_labels, feature.token_type_ids, required_token_type_id=1)

        entities = []
        previous_end_index = None
        for start_index, start_label in enumerate(start_labels):
            if start_label != 1:
                continue

            for end_index, end_label in enumerate(end_labels):
                if end_index >= start_index and start_label == end_label == span_labels[start_index][end_index]:
                    if  remove_overlap == 'short' and end_index == previous_end_index:
                        continue
                    elif  remove_overlap == 'long' and end_index == previous_end_index:
                        entities.pop()

                    entities.append(EntityExample(
                        start_idx=start_index,
                        end_idx=end_index+1,
                        entity=''.join(feature.context_words[start_index: end_index+1]),
                        type=query_type))
                    previous_end_index = end_index
                    break

        example_list.append(NerExample(
            text=''.join(feature.context_words), entities=entities, words=feature.context_words))
        if example_id not in group_example_id_map:
            group_example_id_map[example_id] = []
        group_example_id_map[example_id].append(index)

    for _, group_index_list in sorted(group_example_id_map.items(), key=lambda item: item[0]):
        group_entities = []
        for group_index in group_index_list:
            group_entities.extend(example_list[group_index].entities)
        group_example_list.append(NerExample(
                text=example_list[group_index_list[0]].text,
                words=example_list[group_index_list[0]].text,
                entities=sorted(group_entities, key=lambda entity: entity.start_idx)))


    return group_example_list

