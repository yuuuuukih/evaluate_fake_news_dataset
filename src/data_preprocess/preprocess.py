'''
Process the raw fake news dataset to fine-tuneable json format.
'''
import os
import json
from typing import Literal
from tqdm import tqdm

from type.fake_news_dataset import FakeNewsDataset, DocForDataset
from type.processed_dataset import ProcessedDataset
from const.special_token import TARGET_TOKEN

from data_preprocess.gpt_for_summary import get_summarized_content

'''
mode
- base: no timeline. 1 (is_fake) or 0 for one document.
- pre_target_timeline: Create a timeline with documents that are earlier than the target document.
- all_timeline: Create a timeline with all documents.
'''

class Preprocessor:
    def __init__(self, setting: Literal['rep0', 'rep1','rep2','rep3', 'ins0', 'ins1', 'ins2'], only_short_description: bool = False) -> None:
        self.setting_str = setting.rstrip('0123456789') # rep or ins
        self.only_short_description = only_short_description

        # self.sep_token = '</s>'
        self.sep_token = '[SEP]'
        self.target_token = TARGET_TOKEN

    def load_summarized_dataset(self, dataset_path: str) -> None:
        self.raw_dataset: FakeNewsDataset = self._load_raw_dataset(dataset_path)
        self.timelines_with_summarized_content = self.get_timelines_with_summarized_content(self.raw_dataset)

    def _load_raw_dataset(self, path: str):
        with open(path, 'r') as f:
            raw_dataset: FakeNewsDataset = json.load(f)
        return raw_dataset

    def get_timelines_with_summarized_content(self, raw_dataset: FakeNewsDataset):
        print('=== Start summarizing content ===')
        for timeline in tqdm(raw_dataset['data']):
            if self.setting_str == 'rep':
                replaced_doc = timeline['replaced_doc']
                for _ in range(50):
                    summarized_content = get_summarized_content(replaced_doc['content'])
                    if 150 < len(summarized_content.split()) < 250:
                        replaced_doc['content'] = summarized_content
                        break
                    else:
                        print(f"Summarized content ({len(summarized_content.split())}) is too short or too long. Retrying...")

            for doc in timeline['timeline']:
                if doc['is_fake']:
                    continue
                for _ in range(50):
                    summarized_content = get_summarized_content(doc['content'])
                    if 150 < len(summarized_content.split()) < 250:
                        doc['content'] = summarized_content
                        break
                    else:
                        print(f"Summarized content ({len(summarized_content.split())}) is too short or too long. Retrying...")

            # """ TEST """
            # break
            # """ TEST """

        timelines_with_summarized_content = raw_dataset['data']
        return timelines_with_summarized_content

    def _template_of_src(self, doc: DocForDataset, content: bool = False) -> str:
        if content:
            return f"date: {doc['date']} {self.sep_token} headline: {doc['headline']} {self.sep_token} content: {doc['content']}"
        else:
            return f"date: {doc['date']} {self.sep_token} headline: {doc['headline']} {self.sep_token} short_description: {doc['short_description']}"

    def process(self, mode: Literal['base', 'pre_target_timeline', 'all_timeline'], out_dir: str, json_file_name: str):
        new_dataset: ProcessedDataset = {
            'name': f"{self.raw_dataset['name']} ({mode})",
            'length': 0,
            'setting': self.raw_dataset['setting'],
            'data': []
        }

        if mode == 'base':
            for timeline in self.timelines_with_summarized_content:
                for doc in timeline['timeline']:
                    new_dataset['data'].append({
                        'src': self._template_of_src(doc, content=True),
                        'tgt': int(doc['is_fake']) #fake -> 1, real -> 0
                    })

        elif mode == 'pre_target_timeline' or mode == 'all_timeline':
            for timeline in self.timelines_with_summarized_content:
                for i in range(2, len(timeline['timeline'])-1):
                    # Determine if the i-th document of the timeline is fake or real.
                    tgt = int(timeline['timeline'][i]['is_fake']) #fake -> 1, real -> 0

                    if self.setting_str == 'rep':
                        replaced_doc = timeline['replaced_doc']
                        doc_m2 = self._template_of_src(timeline['timeline'][i-2] if not timeline['timeline'][i-2]['is_fake'] else replaced_doc) # target-2
                        doc_m1 = self._template_of_src(timeline['timeline'][i-1] if not timeline['timeline'][i-1]['is_fake'] else replaced_doc) # target-1
                        doc_t = self._template_of_src(timeline['timeline'][i], content=not self.only_short_description) # target
                        doc_p1 = self._template_of_src(timeline['timeline'][i+1] if not timeline['timeline'][i+1]['is_fake'] else replaced_doc) # target+1
                    elif self.setting_str == 'ins':
                        pass
                    src = f"{doc_m2} {self.sep_token} {self.sep_token} {doc_m1} {self.target_token} {doc_t} {self.target_token}"
                    if mode == 'all_timeline':
                        src += f" {doc_p1}"

                    new_dataset['data'].append({
                        'src': src,
                        'tgt': tgt
                    })

        else:
            raise ValueError(f"Invalid mode: {mode}")

        new_dataset['length'] = len(new_dataset['data'])
        self._save_processed_dataset(out_dir, json_file_name, new_dataset)

    def _save_processed_dataset(self, out_dir: str, json_file_name: str, dataset: ProcessedDataset):
        file_path = os.path.join(out_dir, f"{json_file_name}.json")
        with open(file_path, 'w') as F:
            json.dump(dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))
            print(f'Data is saved to {json_file_name}.json')

    def test_gpt(self):
        input_content = "Scientists from the Ocean Exploration Institute have made an exciting discovery in the depths of the Pacific Ocean. During a recent research expedition, they stumbled upon a previously unknown species of marine life, which has left the scientific community in awe. The expedition, led by Dr. Emily Johnson, set out to explore the uncharted regions of the Pacific Ocean. Equipped with state-of-the-art technology, the team descended to depths of over 3,000 meters, where they encountered a breathtaking sight. Among the vibrant coral reefs and mysterious underwater caves, they discovered a species of fish unlike anything ever seen before. The newly discovered fish, named \"Aurora Fish\" due to its mesmerizing bioluminescent glow, possesses unique characteristics that set it apart from any other known species. Its body is adorned with intricate patterns of neon colors, which emit a soft, ethereal light. This adaptation is believed to aid in communication and attracting prey in the dark depths of the ocean. Dr. Johnson and her team were particularly fascinated by the Aurora Fish's ability to change its color and pattern in response to its surroundings. This remarkable camouflage mechanism allows the fish to blend seamlessly with its environment, making it nearly invisible to predators. Further analysis of the Aurora Fish revealed that it possesses a specialized organ that produces a chemical compound with potential medicinal properties. This discovery has sparked great interest among pharmaceutical companies, as it could lead to the development of new drugs to treat various diseases. The scientists also observed unique social behavior among the Aurora Fish. They discovered that these fish form intricate hierarchies within their schools, with dominant individuals leading and protecting the group. This finding sheds light on the complex social dynamics of marine life and raises questions about the intelligence and social structures of deep-sea creatures. The discovery of the Aurora Fish highlights the importance of exploring and preserving the world's oceans. Dr. Johnson emphasized the need for continued research and conservation efforts to protect these fragile ecosystems and the incredible biodiversity they harbor. The findings of this expedition have been published in the prestigious scientific journal, Marine Biology, and have garnered international attention. Scientists from around the world are now eager to study the Aurora Fish and unravel the mysteries of its unique adaptations. As we continue to explore the vast depths of our oceans, it is clear that there is still so much to discover. The Aurora Fish serves as a reminder of the wonders that await us beneath the surface and the importance of preserving these delicate ecosystems for future generations."
        summarized_content = get_summarized_content(input_content)
        print(len(summarized_content.split()))
        print(summarized_content)
