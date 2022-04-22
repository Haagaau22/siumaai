import torch
from typing import List, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class EntityExample:
    """
    [start_idx: end_idx) 注意尾部开区间
    """
    start_idx: int
    end_idx: int
    entity: str
    type: str


@dataclass
class NerExample:
    text: str
    entities: List[EntityExample]
    words: List[str]


@dataclass
class NerFeature:
    text: str
    entities: List[EntityExample]
    words: List[str]
    word_ids: List[Union[int, None]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor


# @dataclass
# class PairNerExample:
#     query: str
#     context: str
#     entities: List[EntityExample]
#     query_words: List[str]
#     context_words: List[str]
# 
# 
# @dataclass
# class PairNerFeature:
#     word_ids: List[Union[int, None]]
#     input_ids: torch.Tensor
#     attention_mask: torch.Tensor
#     token_type_ids: torch.Tensor


# class NerDataset(Dataset):
#     def __init__(self, example_list: List[NerExample]):
#         self.example_list = example_list
# 
#     def __len__(self):
#         return len(self.example_list)
# 
#     def __getitem__(self, idx) -> NerExample:
#         return self.example_list[idx]
