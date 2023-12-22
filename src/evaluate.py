import os
import torch
from finetune.model_pred_by_cls import BinaryClassifierByCLS
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser

class DatasetEvaluation:
    def __init__(self, model_name, checkpoint_path, mode) -> None:
        self.finetuned_model = BinaryClassifierByCLS.load_from_checkpoint(checkpoint_path, model_name=model_name, add_target_token=False if mode == 'base' else True)
        self.model = self.finetuned_model.model
        self.tokenizer = self.finetuned_model.tokenizer

        self.model.eval()

    def _safe_divide(self, a, b):
        if b == 0:
            return 0
        return a / b

    def evaluate(self, test_data_dir):
        test_dataset = load_dataset('json', data_files={
            'test': os.path.join(test_data_dir, 'test.json')
        }, field='data')['test']

        tp, tn, fp, fn = 0, 0, 0, 0

        with torch.no_grad():
            for example in tqdm(test_dataset):
                src_texts: str = example['src']
                tgt_texts: int = example['tgt']
                # tokenize src text.
                tokenized_src_texts = self.tokenizer(src_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

                input_ids = tokenized_src_texts['input_ids'].to(self.model.device)
                attention_mask = tokenized_src_texts['attention_mask'].to(self.model.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                proba = self.finetuned_model.sigmoid(output.logits).squeeze()
                pred = proba.item()

                if tgt_texts == 1:
                    if pred >= 0.5:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred >= 0.5:
                        fp += 1
                    else:
                        tn += 1

            acc = self._safe_divide(tp+tn, tp+tn+fp+fn)
            precision = self._safe_divide(tp, tp+fp)
            recall = self._safe_divide(tp, tp+fn)
            f1 = self._safe_divide(2*precision*recall, precision+recall)
            print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
            print(f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', default=None, choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument("--root_dir", default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--ckpt_file_name", default='best_cls')
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.root_dir, args.sub_dir, args.mode, f'{args.ckpt_file_name}.ckpt')
    test_data_dir = os.path.join(args.root_dir, args.sub_dir, args.mode)
    print(checkpoint_path)

    de = DatasetEvaluation(args.model_name, checkpoint_path, args.mode)
    de.evaluate(test_data_dir)

if __name__ == '__main__':
    main()