import os
import json
import random

from argparse import ArgumentParser
from type.processed_dataset import ProcessedDataset

class AdditionalPreprocessor:
    def __init__(self, root_dir: str, sub_dir: str) -> None:
        random.seed(42)

        self.root_dir = root_dir
        self.sub_dir = sub_dir

    """
    Align the number of data in base
    """
    def align_the_number_of_data_in_base(self):
        DATA_SIZE = {'train': 607, 'val': 89, 'test': 123}
        for what in ['train', 'val', 'test']:
            base_dataset_path = os.path.join(self.root_dir, self.sub_dir, 'base', f'{what}_full.json')
            with open(base_dataset_path, 'r') as F:
                dataset: ProcessedDataset = json.load(F)
            sampled_dataset = []
            real_news_examples = []
            for example in dataset['data']:
                if example['tgt'] == 1:
                    sampled_dataset.append(example)
                else:
                    real_news_examples.append(example)
            sampled_dataset += random.sample(real_news_examples, DATA_SIZE[what] - len(sampled_dataset)) #len(sampled_dataset)=607
            dataset['data'] = sampled_dataset
            dataset['length'] = len(sampled_dataset)
            with open(os.path.join(self.root_dir, self.sub_dir, 'base', f'{what}.json'), 'w') as F:
                json.dump(dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))

    """
    Split training data into 25%, 50%, and 75% sizes
    """
    def split_training_data_into_25_50_75(self):
        for ratio in [0.25, 0.5, 0.75, 1]:
            ratio_str = str(int(ratio * 100))
            for mode in ['base', 'pre_target_timeline', 'all_timeline']:
                train_data_path = os.path.join(self.root_dir, self.sub_dir, mode, 'train.json')
                with open(train_data_path, 'r') as F:
                    train_dataset: ProcessedDataset = json.load(F)
                sampled_train_dataset = random.sample(train_dataset['data'], int(len(train_dataset['data']) * ratio)) #len(train_dataset['data'])=607
                train_dataset['data'] = sampled_train_dataset
                train_dataset['length'] = len(sampled_train_dataset)
                with open(os.path.join(self.root_dir, self.sub_dir, mode, f'train_{ratio_str}.json'), 'w') as F:
                    json.dump(train_dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))

    """
    Create new pattern dataset (1 context - 1 target - 1 future) from all_timeline (2 context - 1 target - 1 future)
    """
    def remove_until_double_sep(self, text, sep="[SEP]"):
        # セパレータが連続して現れる部分を検出
        double_sep = sep + " " + sep
        # double_sepの位置を見つける
        double_sep_pos = text.find(double_sep)
        if double_sep_pos == -1:
            # もし見つからなければ、オリジナルのテキストを返す
            return text
        else:
            # double_sepの後ろの部分を返す
            return text[double_sep_pos + len(double_sep) + 1:]  # +1はスペースの分

    def create_new_pattern_dataset(self):
        new_pattern = "around_target"
        new_dir_path = os.path.join(self.root_dir, self.sub_dir, new_pattern)
        all_timeline_dir_path = os.path.join(self.root_dir, self.sub_dir, "all_timeline")

        os.makedirs(new_dir_path, exist_ok=True)
        for what in ['train_25', 'train_50', 'train_75', 'train_100', 'val', 'test']:
            file_path = os.path.join(all_timeline_dir_path, f'{what}.json')
            with open(file_path, 'r') as F:
                dataset: ProcessedDataset = json.load(F)

            new_dataset = {
                'name': dataset['name'],
                'length': dataset['length'],
                'setting': dataset['setting'],
                'data': []
            }

            for example in dataset['data']:
                src_texts: str = example['src']
                tgt_texts: int = example['tgt']
                new_src_texts = self.remove_until_double_sep(src_texts)
                new_example = {
                    'src': new_src_texts,
                    'tgt': tgt_texts
                }
                new_dataset['data'].append(new_example)

            with open(os.path.join(new_dir_path, f'{what}.json'), 'w') as F:
                json.dump(new_dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))

    """
    Remove phrase beginning with "In" in the fake news content in the preprocessed dataset.
    """
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    args = parser.parse_args()

    ap = AdditionalPreprocessor(root_dir=args.root_dir, sub_dir=args.sub_dir)

    """
    Align the number of data in base
    """
    # ap.align_the_number_of_data_in_base()


    """
    Split training data into 25%, 50%, and 75% sizes
    """
    # ap.split_training_data_into_25_50_75()


    """
    Create new pattern dataset (1 context - 1 target - 1 future) from all_timeline (2 context - 1 target - 1 future)
    """
    ap.create_new_pattern_dataset()

    """
    Remove phrase beginning with "In" in the fake news content in the preprocessed dataset.
    """



if __name__ == '__main__':
    main()