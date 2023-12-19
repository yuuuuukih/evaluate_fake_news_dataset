import os
from data_preprocess.preprocess import Preprocessor

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'timeline_aware'])
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='')
    args = parser.parse_args()

    for what in ['train', 'val', 'test']:
        dataset_path = os.path.join(args.root_dir, args.sub_dir, f'{what}.json')
        out_dir = os.path.join(args.root_dir, args.sub_dir, f'{args.mode}')
        os.makedirs(out_dir, exist_ok=True)
        pp = Preprocessor(args.mode)
        pp.process(dataset_path, out_dir, what)

if __name__ == '__main__':
    main()
