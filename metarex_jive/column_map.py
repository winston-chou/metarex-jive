from dataclasses import dataclass
from typing import List


@dataclass
class ColumnMap:
    test_id: str
    treatment_id: str
    outcome: str
    mediators: List[str]
