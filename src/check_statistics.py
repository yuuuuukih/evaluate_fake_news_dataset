import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def calculate_boxplot_data(data: list, whis: float = 1.5):
    """箱ひげ図に必要な統計値（平均値含む）を計算する関数"""
    min_val = np.min(data)  # 最小値
    max_val = np.max(data)  # 最大値
    q1 = np.percentile(data, 25)  # 第一四分位数 (Q1)
    q3 = np.percentile(data, 75)  # 第三四分位数 (Q3)
    median = np.median(data)  # 中央値
    mean = np.mean(data)  # 平均値

    iqr = q3 - q1  # 四分位範囲 (IQR)
    lower_bound = q1 - (whis * iqr)  # 外れ値に対する下限界
    upper_bound = q3 + (whis * iqr)  # 外れ値に対する上限界

    # 外れ値を除いた実際の最小値と最大値
    actual_min_val = min([x for x in data if x >= lower_bound], default=min_val)
    actual_max_val = max([x for x in data if x <= upper_bound], default=max_val)

    return {
        'min': min_val,
        'max': max_val,
        'q1': q1,
        'q3': q3,
        'median': median,
        'mean': mean,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'actual_min_val': actual_min_val,
        'actual_max_val': actual_max_val,
        'outliers': [x for x in data if x < lower_bound or x > upper_bound]
    }

def plot_boxplot(data: list, label: list[str] = ['real (before)', 'real (after 0-shot)', 'real (after few-shot)', 'fake'], file_name: str = 'boxplot', whis: float = 1.5):
    """箱ひげ図を描画する関数"""
    fig = plt.figure(figsize=(8, 6))
    # ax.set_title('The box plot of the data')

    # グリッドの設定
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 3])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.boxplot([data[0]], showmeans=True, whis=whis)
    ax2.boxplot(data[1:], showmeans=True, whis=whis)
    ax1.set_xticklabels([label[0]])
    ax2.set_xticklabels(label[1:])
    ax1.set_ylabel('The number of words of the content.')
    fig.savefig(file_name)

def get_content_text(src: str):
    content_keyword = 'content:'
    start_index = src.find(content_keyword)

    if start_index != -1:
        # 'content:' の文字列の長さを加算して開始位置を得る
        content_start = start_index + len(content_keyword)
        # 開始位置から末尾までを抽出
        content = src[content_start:].strip()

        return content
    else:
        print("The specified keyword cannot be found.")

def main():
    before_sammarized_data_path = '/mnt/mint/hara/datasets/news_category_dataset/dataset/diff7/timeline_diff7.json'
    data_dir = '/mnt/mint/hara/datasets/news_category_dataset/dataset/diff7_rep3/base/'

    few_shot_sample_path = '/mnt/mint/hara/datasets/news_category_dataset/dataset/diff7_rep3/sampled_summarized.json'

    before_sammarized_real_doc_lens = []
    real_doc_lens = []
    fake_doc_lens = []

    few_shot_real_doc_lens = []

    with open(before_sammarized_data_path, 'r') as F:
        before_sammarized_data = json.load(F)

    with open(few_shot_sample_path, 'r') as F:
        few_shot_sample = json.load(F)

    manage_doc_ids_for_before_summarized = []
    for keyword_group in before_sammarized_data['data']:
        for timeline in keyword_group['timeline_info']['data']:
            for doc in timeline['timeline']:
                doc_id: int = doc['ID']
                doc_content: str = doc['content']
                doc_len = len(doc_content.split())
                if doc_id not in set(manage_doc_ids_for_before_summarized):
                    manage_doc_ids_for_before_summarized.append(doc_id)
                    before_sammarized_real_doc_lens.append(doc_len)

    manage_doc_ids_for_few_shot_summarized = []
    for timeline_data in few_shot_sample:
        few_shot_real_doc_lens.append(len(timeline_data['replaced_doc']['content'].split()))
        for doc in timeline_data['timeline']:
            doc_id: int = doc['ID']
            doc_content: str = doc['content']
            doc_len = len(doc_content.split())
            if doc_id not in set(manage_doc_ids_for_few_shot_summarized):
                manage_doc_ids_for_few_shot_summarized.append(doc_id)
                few_shot_real_doc_lens.append(doc_len)

    for what in ['train', 'val', 'test']:
        with open(os.path.join(data_dir, f'{what}.json'), 'r') as F:
            what_json = json.load(F)

        for example in what_json['data']:
            content_len = len(get_content_text(example['src']).split())
            if example['tgt'] == 1:
                fake_doc_lens.append(content_len)
            else:
                real_doc_lens.append(content_len)

    print(calculate_boxplot_data(before_sammarized_real_doc_lens))
    print(calculate_boxplot_data(real_doc_lens))
    print(calculate_boxplot_data(fake_doc_lens))
    print(calculate_boxplot_data(few_shot_real_doc_lens))
    plot_boxplot([before_sammarized_real_doc_lens, real_doc_lens, few_shot_real_doc_lens, fake_doc_lens])

if __name__ == '__main__':
    main()
