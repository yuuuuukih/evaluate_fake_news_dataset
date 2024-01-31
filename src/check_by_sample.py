import os
import json
import random

def main():
    data_dir = '/mnt/mint/hara/datasets/news_category_dataset/dataset/'
    sub_dir = 'diff7_rep3/base'
    file_name = 'sample50x2'

    #concat_data = []
    concat_data_fake = []
    concat_data_real = []
    for what in ['train_100', 'val', 'test']:
        with open(os.path.join(data_dir, sub_dir, f'{what}.json'), 'r') as F:
            what_json = json.load(F)
        # concat_data.extend(what_json['data'])
        for example in what_json['data']:
            if example['tgt'] == 1:
                concat_data_fake.append(example)
            else:
                concat_data_real.append(example)

    #print(len(concat_data))
    # sampled_train_json = random.sample(concat_data, 20)
    sampled_fake_json = random.sample(concat_data_fake, 50)
    sampled_real_json = random.sample(concat_data_real, 50)
    sampled_json = sampled_fake_json + sampled_real_json

    with open(os.path.join(data_dir, sub_dir, f'{file_name}.json'), 'w') as F:
        json.dump(sampled_json, F, indent=4, ensure_ascii=False, separators=(',', ': '))

if __name__ == '__main__':
    main()