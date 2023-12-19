from typing import TypedDict, Literal

class OneCombination(TypedDict):
    src: str
    tgt: str

class ProcessedDataset(TypedDict):
    name: str
    length: int
    setting: Literal['none', 'rep0', 'rep1','rep2','rep3', 'ins0', 'ins1', 'ins2']
    data: list[OneCombination]