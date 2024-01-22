import os
import re
import json
from data_preprocess.preprocess import Preprocessor
from utils.measure_exe_time import measure_exe_time

from argparse import ArgumentParser

@measure_exe_time
def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='diff7_rep3', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    args = parser.parse_args()

    setting = re.search(r'_(\w+)$', args.sub_dir).group(1)

    sample_json_file_name = 'sample20_0109.json'
    dataset_path = os.path.join(args.root_dir, args.sub_dir, sample_json_file_name)
    with open(dataset_path, 'r') as F:
        sample_json = json.load(F)
    sample_json['data'] = sample_json['data'][21:22]

    pp = Preprocessor(setting=setting, only_short_description=False)
    summarized_sample_json = pp.get_timelines_with_summarized_content(sample_json)

    with open(os.path.join(args.root_dir, args.sub_dir, 'sampled_summarized.json'), 'w') as F:
        json.dump(summarized_sample_json, F, indent=4, ensure_ascii=False, separators=(',', ': '))

if __name__ == '__main__':
    main()
