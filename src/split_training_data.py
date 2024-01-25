import os
import json
import random

from argparse import ArgumentParser
from type.processed_dataset import ProcessedDataset

def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    args = parser.parse_args()

    """
    Align the number of data in base
    """
    DATA_SIZE = {'train': 607, 'val': 89, 'test': 123}
    for what in ['train', 'val', 'test']:
        base_dataset_path = os.path.join(args.root_dir, args.sub_dir, 'base', f'{what}_full.json')
        with open(base_dataset_path, 'r') as F:
            dataset: ProcessedDataset = json.load(F)
        sampled_dataset = random.sample(dataset['data'], DATA_SIZE[what])
        dataset['data'] = sampled_dataset
        dataset['length'] = len(sampled_dataset)
        with open(os.path.join(args.root_dir, args.sub_dir, 'base', f'{what}.json'), 'w') as F:
            json.dump(dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))

    """
    Split training data into 25%, 50%, and 75% sizes
    """
    for ratio in [0.25, 0.5, 0.75, 1]:
        ratio_str = str(int(ratio * 100))
        for mode in ['base', 'pre_target_timeline', 'all_timeline']:
            train_data_path = os.path.join(args.root_dir, args.sub_dir, mode, 'train.json')
            with open(train_data_path, 'r') as F:
                train_dataset: ProcessedDataset = json.load(F)
            sampled_train_dataset = random.sample(train_dataset['data'], int(len(train_dataset['data']) * ratio)) #len(train_dataset['data'])=607
            train_dataset['data'] = sampled_train_dataset
            train_dataset['length'] = len(sampled_train_dataset)
            with open(os.path.join(args.root_dir, args.sub_dir, mode, f'train_{ratio_str}.json'), 'w') as F:
                json.dump(train_dataset, F, indent=4, ensure_ascii=False, separators=(',', ': '))


if __name__ == '__main__':
    main()