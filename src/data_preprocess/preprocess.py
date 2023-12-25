'''
Process the raw fake news dataset to fine-tuneable json format.
'''
import os
import json
from typing import Literal

from type.fake_news_dataset import FakeNewsDataset, DocForDataset
from type.processed_dataset import ProcessedDataset
from special_token import TARGET_TOKEN

'''
mode
- base: no timeline. 1 (is_fake) or 0 for one document.
- pre_target_timeline: Create a timeline with documents that are earlier than the target document.
- all_timeline: Create a timeline with all documents.
'''

class Preprocessor:
    def __init__(self, mode: Literal['base', 'pre_target_timeline', 'all_timeline'], only_short_description: bool = False) -> None:
        self.mode = mode
        self.only_short_description = only_short_description

        # self.sep_token = '</s>'
        self.sep_token = '[SEP]'
        self.target_token = TARGET_TOKEN

    def _load_raw_dataset(self, path: str):
        with open(path, 'r') as f:
            raw_dataset: FakeNewsDataset = json.load(f)
        return raw_dataset

    def _template_of_src(self, doc: DocForDataset, content: bool = False) -> str:
        if content:
            return f"date: {doc['date']} {self.sep_token} headline: {doc['headline']} {self.sep_token} content: {doc['content']}"
        else:
            return f"date: {doc['date']} {self.sep_token} headline: {doc['headline']} {self.sep_token} short_description: {doc['short_description']}"

    def process(self, dataset_path: str, out_dir: str, json_file_name: str):
        raw_dataset = self._load_raw_dataset(dataset_path)
        new_dataset: ProcessedDataset = {
            'name': f"{raw_dataset['name']} ({self.mode})",
            'length': 0,
            'setting': raw_dataset['setting'],
            'data': []
        }

        if self.mode == 'base':
            for timeline in raw_dataset['data']:
                for doc in timeline['timeline']:
                    new_dataset['data'].append({
                        'src': self._template_of_src(doc, content=True),
                        'tgt': int(doc['is_fake']) #fake -> 1, real -> 0
                    })

        elif self.mode == 'pre_target_timeline' or self.mode == 'all_timeline':
            for timeline in raw_dataset['data']:
                for i in range(len(timeline['timeline'])):
                    # Determine if the i-th document of the timeline is fake or real.
                    src = ''
                    tgt = int(timeline['timeline'][i]['is_fake']) #fake -> 1, real -> 0
                    for j, doc in enumerate(timeline['timeline']):
                        if i == j:
                            src += f"{self.target_token} {self._template_of_src(doc, content=not self.only_short_description)} {self.target_token} "
                            # If the mode is 'pre_target_timeline', we don't need to add the following documents.
                            if self.mode == 'pre_target_timeline':
                                break
                        elif i == j+1 or j == len(timeline['timeline'])-1:
                            src += f"{self._template_of_src(doc)} "
                        else:
                            src += f"{self._template_of_src(doc)} {self.sep_token} {self.sep_token} "
                    new_dataset['data'].append({
                        'src': src,
                        'tgt': tgt
                    })

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        new_dataset['length'] = len(new_dataset['data'])
        self._save_processed_dataset(out_dir, json_file_name, new_dataset)

    def _save_processed_dataset(self, out_dir: str, json_file_name: str, dataset: ProcessedDataset):
        file_path = os.path.join(out_dir, f"{json_file_name}.json")
        with open(file_path, 'w') as F:
            json.dump(dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))
            print(f'Data is saved to {json_file_name}.json')
