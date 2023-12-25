import os
from data_preprocess.preprocess import Preprocessor

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument('--only_sd', default=False, action='store_true', help='only short description')
    args = parser.parse_args()

    for what in ['train', 'val', 'test']:
        dataset_path = os.path.join(args.root_dir, args.sub_dir, f'{what}.json')

        if not args.only_sd:
            out_dir = os.path.join(args.root_dir, args.sub_dir, f'{args.mode}')
        else:
            out_dir = os.path.join(args.root_dir, args.sub_dir, 'only_short_description', f'{args.mode}')
        os.makedirs(out_dir, exist_ok=True)

        pp = Preprocessor(args.mode, only_short_description=args.only_sd)
        pp.process(dataset_path, out_dir, what)

if __name__ == '__main__':
    main()
