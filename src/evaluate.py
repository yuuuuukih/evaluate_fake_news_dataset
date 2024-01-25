import os
import torch
from finetune.model_pred_by_cls import BinaryClassifierByCLS
from finetune.model_pred_by_target import BinaryClassifierByTARGET
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import default_data_collator

class DatasetEvaluation:
    def __init__(
            self,
            model_name: str,
            checkpoint_path: str,
            mode: Literal['base', 'pre_target_timeline', 'all_timeline'],
            pred_by: Literal['cls', 'target'],
            concat_or_mean: Literal['concat', 'mean']
            ) -> None:
        if pred_by == 'cls':
            self.finetuned_model = BinaryClassifierByCLS.load_from_checkpoint(checkpoint_path, model_name=model_name, add_target_token=False if mode == 'base' else True)
        elif pred_by == 'target':
            self.finetuned_model = BinaryClassifierByTARGET.load_from_checkpoint(checkpoint_path, model_name=model_name, concat_or_mean=concat_or_mean)
            self.finetuned_model.set_concat_or_mean(concat_or_mean)
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
                if self.finetuned_model.__class__.__name__ == 'BinaryClassifierByCLS':
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    proba = self.finetuned_model.sigmoid(output.logits).squeeze()
                    # proba = self.finetuned_model.forward(input_ids=input_ids, attention_mask=attention_mask)
                    pred = proba.item()
                elif self.finetuned_model.__class__.__name__ == 'BinaryClassifierByTARGET':
                    loggits = self.finetuned_model.forward(input_ids=input_ids, attention_mask=attention_mask)
                    sigmoid = torch.nn.Sigmoid()
                    preba = sigmoid(loggits).squeeze()
                    pred = preba.item()

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
    parser.add_argument('--pred_by', default='target', choices=['cls', 'target'])
    parser.add_argument('--concat_or_mean', default='concat', choices=['concat', 'mean'])
    parser.add_argument('--mode', default=None, choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument("--root_dir", default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--ckpt_file_name", default='default')
    args = parser.parse_args()

    # ckpt_file_name
    if args.ckpt_file_name == 'default':
        ckpt_file_name = f"best_target_{args.concat_or_mean}" if args.pred_by == 'target' else 'best_cls'
    else:
        ckpt_file_name = args.ckpt_file_name
    checkpoint_path = os.path.join(args.root_dir, args.sub_dir, args.mode, f'{ckpt_file_name}.ckpt')
    test_data_dir = os.path.join(args.root_dir, args.sub_dir, args.mode)
    print(checkpoint_path)

    de = DatasetEvaluation(args.model_name, checkpoint_path, args.mode, args.pred_by, args.concat_or_mean)
    de.evaluate(test_data_dir)

if __name__ == '__main__':
    main()