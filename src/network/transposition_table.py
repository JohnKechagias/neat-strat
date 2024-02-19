import sys
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional

from .constants import Move


class NodeFlag(IntEnum):
    EXACT = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()


@dataclass(frozen=True)
class TranspositionEntry:
    key: int
    score: float
    best_move: Optional[Move]
    depth: int
    flag: NodeFlag


class TranspositionTable:
    def __init__(self, sizeMB: int):
        size_of_entry = sys.getsizeof(TranspositionEntry)
        desired_table_size_in_bytes = sizeMB * 1024 * 1024
        num_of_entries = desired_table_size_in_bytes // size_of_entry
        self.max_entries_count = num_of_entries
        self.table: list[Optional[TranspositionEntry]] = [
            None for _ in range(num_of_entries)
        ]

    def add(
        self,
        key: int,
        value: float,
        best_move: Optional[Move],
        depth: int,
        flag: NodeFlag,
    ):
        entry = TranspositionEntry(key, value, best_move, depth, flag)
        index = self._get_index_from_zobri_key(entry.key)
        self.table.insert(index, entry)

    def get(self, zobri_key: int) -> Optional[TranspositionEntry]:
        index = self._get_index_from_zobri_key(zobri_key)
        entry = self.table[index]

        if entry is None or entry.key != zobri_key:
            return None

        return entry

    def clear(self):
        self.table.clear()

    def _get_index_from_zobri_key(self, zobri_key: int):
        return zobri_key % self.max_entries_count


def evaluate_entry(
    entry: Optional[TranspositionEntry],
    depth: int,
    alpha: float,
    beta: float,
) -> tuple[float, bool, Optional[Move]]:
    adjusted_score = 0.0
    should_use = False
    best_move: Optional[Move] = None

    if entry is None:
        return adjusted_score, should_use, best_move

    best_move = entry.best_move

    if entry.depth < depth:
        return adjusted_score, should_use, best_move

    score = entry.score
    if entry.flag == NodeFlag.EXACT:
        adjusted_score = score
        should_use = True

    if entry.flag == NodeFlag.UPPER_BOUND and score <= alpha:
        adjusted_score = alpha
        should_use = True

    if entry.flag == NodeFlag.LOWER_BOUND and score >= beta:
        adjusted_score = beta
        should_use = True

    return adjusted_score, should_use, best_move
