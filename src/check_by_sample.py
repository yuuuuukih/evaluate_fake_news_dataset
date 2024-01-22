import os
import json
import random

def main():
    data_dir = '/mnt/mint/hara/datasets/news_category_dataset/dataset/'
    sub_dir = 'filtered_diff7_rep3'
    file_name = 'sample20'

    concat_data = []
    for what in ['train', 'val', 'test']:
        with open(os.path.join(data_dir, sub_dir, f'{what}.json'), 'r') as F:
            what_json = json.load(F)
        concat_data.extend(what_json['data'])

    print(len(concat_data))
    sampled_train_json = random.sample(concat_data, 20)

    with open(os.path.join(data_dir, sub_dir, f'{file_name}.json'), 'w') as F:
        json.dump(sampled_train_json, F, indent=4, ensure_ascii=False, separators=(',', ': '))

if __name__ == '__main__':
    main()