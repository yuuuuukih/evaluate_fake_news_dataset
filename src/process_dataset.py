import os
import re
from data_preprocess.preprocess import Preprocessor

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument('--only_sd', default=False, action='store_true', help='only short description')
    args = parser.parse_args()

    setting = re.search(r'_(\w+)$', args.sub_dir).group(1)

    for what in ['train', 'val', 'test']:
        dataset_path = os.path.join(args.root_dir, args.sub_dir, f'{what}.json')

        if not args.only_sd:
            out_dir_base = os.path.join(args.root_dir, args.sub_dir, 'base')
            out_dir_pre_target_timeline = os.path.join(args.root_dir, args.sub_dir, 'pre_target_timeline')
            out_dir_all_timeline = os.path.join(args.root_dir, args.sub_dir, 'all_timeline')
        else:
            out_dir_base = os.path.join(args.root_dir, args.sub_dir, 'only_short_description', 'base')
            out_dir_pre_target_timeline = os.path.join(args.root_dir, args.sub_dir, 'only_short_description', 'pre_target_timeline')
            out_dir_all_timeline = os.path.join(args.root_dir, args.sub_dir, 'only_short_description', 'all_timeline')
        os.makedirs(out_dir_base, exist_ok=True)
        os.makedirs(out_dir_pre_target_timeline, exist_ok=True)
        os.makedirs(out_dir_all_timeline, exist_ok=True)

        pp = Preprocessor(setting=setting, only_short_description=args.only_sd)
        pp.load_summarized_dataset(dataset_path)
        pp.process('base', out_dir_base, what)
        pp.process('pre_target_timeline', out_dir_pre_target_timeline, what)
        pp.process('all_timeline', out_dir_all_timeline, what)

if __name__ == '__main__':
    main()
