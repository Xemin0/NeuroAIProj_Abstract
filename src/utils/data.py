from math import factorial
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from itertools import permutations, combinations
from typing import Optional, Iterable, Set, Union
import warnings
import torch
from torch.utils.data import Dataset, DataLoader, Subset


def powerset_generator(n: int):
    '''Generates each subset represented as indicator vectors'''
    # Iterate through each subset size from 0 to n
    for r in range(n + 1):
        for subset_indices in combinations(range(n), r):
            bit_string = torch.zeros(n, dtype=torch.int)
            bit_string[list(subset_indices)] = 1 
            yield bit_string

@dataclass
class Item:
    value: list
    rank: int
    chain_id: Optional[int] = None
    unique_id: Optional[int] = None

    def __post_init__(self):
        if self.chain_id is not None:
            self.chain_indices = set([self.chain_id])
        else:
            self.chain_id = 0
            self.chain_indices = set([0])

        if self.unique_id is None:
            self.unique_id = self.rank

    @staticmethod
    def decimal_from_bits(bits: torch.tensor):
        # index will be the decimal representation of the bit string
        bitstr = "".join(str(b) for b in bits.tolist())
        return int(bitstr, 2)

    def _same_chain(self, other):
        return self.chain_id == other.chain_id

    def __eq__(self, other):
        return torch.eq(self.value, other.value)

    def __gt__(self, other):
        assert self._same_chain(other), \
          "Items are not comparable! They must belong to the same chain."
        return self.rank > other.rank

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    @classmethod
    def rank_from_feature_repr(cls, feat:torch.tensor):
        '''Assumes feature representation is one-hot'''
        raise NotImplementedError


class Task:
    def __init__(self,
                 task_name: str,
                 ground_set: list[torch.tensor],
                 ranking: tuple = None,
                 target_rule: callable = None):

        task_name = task_name.lower()
        assert "ti" in task_name or "si" in task_name
        self.task_name = task_name
        self.ground_set = ground_set
        self.tree = None
        
        if ranking is not None and "si" in task_name:
            warnings.warn("Explicit rankings should not be given to SI task")
        self.ranking = ranking

        if target_rule is None:
            self.target_rule = lambda a,b: a > b
        else:
            self.target_rule = target_rule

    def construct_tree(self):
        raise NotImplementedError

    def initialize_itemset(self):
        if "ti" in self.task_name:
            return [Item(self.ground_set[i], rank)
                    for i, rank in zip(range(len(self.ground_set)),
                                       self.ranking)]
        raise NotImplementedError

    def master_dataset(self, itemset: list[Item]) -> Dataset:
        return PosetDataset(self.target_rule, itemset=itemset)



class PosetDataset(Dataset):
    def __init__(self,
                 target_rule: callable,
                 itemset: list[Item] = None,
                 chains: list[list[Item]] = None):
        '''
        sample_indices: unique indices for all possible inputs (train+test)
        unique_pairs_iter: iterator over all possible inputs
        '''
        assert not (itemset is None and chains is None), \
            "Either a set of items or a set of chains should be passed in, and not both!"

        if chains is not None:
            raise NotImplementedError

        if itemset is not None:
            self.itemset = sorted(itemset)

        self.target_rule = target_rule
        self._setup_data_structures()
        self.adj_pairs, self.nonadj_pairs = None, None

    def _setup_data_structures(self):
        self.unique_pairs = list(permutations(self.itemset, 2))
        self.idx_to_pair = {idx : (left.unique_id, right.unique_id)
                            for idx, (left, right) in enumerate(self.unique_pairs)}
        self.pair_to_idx = {(item1, item2): idx for idx, (item1, item2) in self.idx_to_pair.items()}

    def types_of_pairs_partition(self) -> tuple[list[int], list[int]]:
        '''Get adjacent/non-adjacent pairs which are encoded with its unique index.'''
        if self.adj_pairs is not None and self.nonadj_pairs is not None:
            return self.adj_pairs, self.nonadj_pairs

        adj, nonadj = [],[]
        for idx, (item1, item2) in enumerate(self.unique_pairs):
            adjacent = item1.rank + 1 == item2.rank or item1.rank - 1 == item2.rank
            if adjacent:
                adj.append(idx)
            else:
                nonadj.append(idx)

        self.adj_pairs, self.nonadj_pairs = adj, nonadj
        return adj, nonadj

    def __len__(self):
        return len(self.idx_to_pair)


    def __getitem__(self, idx): 
        left, right = self.unique_pairs[idx]
        label = torch.tensor(self.target_rule(left, right), dtype=torch.int)
        left = torch.tensor(left.value, dtype=torch.float64)
        right = torch.tensor(right.value, dtype=torch.float64)

        return torch.cat([left, right]), label

    @property
    def train_indices(self) -> list[int]:
        if self.adj_pairs is not None:
            return self.adj_pairs
        adj_pairs, _ = self.types_of_pairs_partition()
        return adj_pairs

    @property
    def test_indices(self) -> list[int]:
        if self.nonadj_pairs is not None:
            return self.nonadj_pairs
        _, nonadj_pairs = self.types_of_pairs_partition()
        return nonadj_pairs
