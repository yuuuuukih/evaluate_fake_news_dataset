'''
This type is from https://github.com/YuukiHaraProgramming/news_timeline/blob/main/src/create_dataset/type/fake_news_dataset.py
'''
from typing import TypedDict, Literal

class Analytics(TypedDict):
    docs_num_in_1timeline: dict[str, int]
    re_execution_num: dict[str, int]
    no_timeline_entity_id: list[int]

class Temperature(TypedDict):
    _1st_response: float # 1st_response
    _2nd_response: float # 2nd_response

class DocsNumIn1timeline(TypedDict):
    min: int
    max: int

class RougeParms(TypedDict):
    rouge_used: bool
    alpha: float
    th_1: float
    th_2: float
    th_l: float

class Setting(TypedDict):
    model: str
    temperature: Temperature
    docs_num_in_1timeline: DocsNumIn1timeline
    top_tl: float
    max_reexe_num: int
    rouge: RougeParms

# -------------------

class DocForDataset(TypedDict):
    ID: int
    is_fake: bool
    headline: str
    short_description: str
    date: str
    content: str

class TimelineDataInfo(TypedDict):
    entity_id: int
    entity_items: list[str]
    timeline: list[DocForDataset]

class NoFakeTimelinesInfo(TypedDict):
    entities_num: int
    setting: Setting
    analytics: Analytics

class FakeNewsDataset(TypedDict):
    name: str
    description: str
    setting: Literal['none', 'rep0', 'rep1','rep2','rep3', 'ins0', 'ins1', 'ins2']
    docs_num_in_1_timeline: dict[str, int]
    no_fake_timelines_info: NoFakeTimelinesInfo
    data: list[TimelineDataInfo]