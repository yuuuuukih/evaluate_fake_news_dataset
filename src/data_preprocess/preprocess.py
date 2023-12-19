'''
Process the raw fake news dataset to fine-tuneable json format.
'''
import os
import json
from typing import Literal
from argparse import ArgumentParser

from type.fake_news_dataset import FakeNewsDataset, DocForDataset
from type.processed_dataset import ProcessedDataset

class Preprocessor:
    def __init__(self, mode: Literal['base', 'timeline_aware']) -> None:
        self.mode = mode

        self.sep_token = '</s>'
        self.target_token = '<target>'

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

        elif self.mode == 'timeline_aware':
            for timeline in raw_dataset['data']:
                for i in range(len(timeline['timeline'])):
                    # Determine if the i-th document of the timeline is fake or real.
                    src = ''
                    tgt = int(timeline['timeline'][i]['is_fake'])
                    for j, doc in enumerate(timeline['timeline']):
                        if i == j:
                            src += f"{self.target_token} {self._template_of_src(doc, content=True)} {self.target_token} "
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
