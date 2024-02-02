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

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients

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

        with torch.set_grad_enabled(True):
            for example in tqdm(test_dataset):
                src_texts: str = example['src']
                tgt_texts: int = example['tgt']
                # tokenize src text.
                tokenized_src_texts = self.tokenizer(src_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

                input_ids = tokenized_src_texts['input_ids'].to(self.model.device)
                attention_mask = tokenized_src_texts['attention_mask'].to(self.model.device)
                token_type_ids = tokenized_src_texts['token_type_ids'].to(self.model.device)
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

                # ig = IntegratedGradients(self.model)
                # attributions_ig = ig.attribute(input_ids, target=tgt_texts, n_steps=500)
                # # visualize
                # vis_data_records_ig = [visualization.VisualizationDataRecord(
                #     attributions_ig,
                #     pred,
                #     tgt_texts,
                #     str(tgt_texts),
                #     attributions_ig.sum(),
                #     tokenized_src_texts['input_ids'],
                #     1
                # )]

                # print('Visualize attributions based on Integrated Gradients')
                # _ = visualization.visualize_text(vis_data_records_ig)
                batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': tgt_texts, 'token_type_ids': token_type_ids}
                self.compute_lig(batch, target_label=torch.as_tensor([batch["labels"]]).to(self.model.device))

            acc = self._safe_divide(tp+tn, tp+tn+fp+fn)
            precision = self._safe_divide(tp, tp+fp)
            recall = self._safe_divide(tp, tp+fn)
            f1 = self._safe_divide(2*precision*recall, precision+recall)
            print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
            print(f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")


    def compute_lig(
            self,
            input_tensor: dict[str, torch.Tensor],
            target_label: torch.Tensor,  # (b, seq)
        ):

        self.lig = LayerIntegratedGradients(self.finetuned_model.forward, self.finetuned_model.model.bert.embeddings)
        self.lig_counter = 0
        breakpoint()
        attributions, delta = self.lig.attribute(inputs=input_tensor["input_ids"],  # (1, seq)
                                                # baselines=torch.zeros_like(input_tensor),
                                                # baselines=input_tensor["input_ids"][:1], # only for testing! replace here
                                                target=target_label,  # (1)
                                                additional_forward_args=(input_tensor["attention_mask"], input_tensor["token_type_ids"]),
                                                return_convergence_delta=True)

        # visualization = viz.visualize_text(
        #     [attributions.squeeze().tolist()],
        #     input_tensor,
        #     [" ".join(self.tokenizer.convert_ids_to_tokens(input_tensor))],
        #     show_colorbar=True,
        # )
        self.lig_counter += 1

        pred: torch.Tensor = self.finetuned_model.forward(
            input_tensor["input_ids"], input_tensor["attention_mask"], input_tensor["token_type_ids"]
        )  # (1, 2)
        # calculate argmax
        score = torch.softmax(pred, dim=1)[0]  # (2)
        attributions_sum = self._summarize_attributions(attributions)

        true_label = target_label[0].item()
        pred_label = torch.argmax(score, dim=0).item()
        # storing couple samples in an array for visualization purposes
        score_vis_record = viz.VisualizationDataRecord(
                                                attributions_sum,
                                                score[0], # pred probability
                                                pred_label, # predicted label (Tensor)
                                                true_label, # true label (int)
                                                pred_label,
                                                # target_label[0].item(), # attribute class?????
                                                # self.tokenizer.convert_ids_to_tokens(input_tensor["input_ids"][0]),
                                                attributions_sum.sum(),
                                                self.tokenizer.convert_ids_to_tokens(input_tensor["input_ids"]),
                                                delta)

        output_path = "/mnt/mint/hara/datasets/news_category_dataset/captum"
        os.makedirs(output_path, exist_ok=True)

        # save the figure
        vis_html = viz.visualize_text(
            [score_vis_record]
        )

        # Save HTML content to a file
        with open(output_path + "score_vis_"+ str(self.lig_counter)+f"_t{true_label}_p{pred_label}.html", "w") as file:
            file.write(vis_html.data)

    @staticmethod
    def _summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

def main():
    parser = ArgumentParser()
    parser.add_argument('--pred_by', default='target', choices=['cls', 'target'])
    parser.add_argument('--concat_or_mean', default='concat', choices=['concat', 'mean'])
    parser.add_argument('--mode', default=None, choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument("--root_dir", default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--ckpt_file_name", default='default')
    parser.add_argument("--split_ratio", default='100', choices=['25', '50', '75', '100'])
    parser.add_argument('--no_in', default=False, action='store_true')
    args = parser.parse_args()

    # ckpt_file_name
    if args.ckpt_file_name == 'default':
        ckpt_file_name = f"best_target_{args.concat_or_mean}_{args.split_ratio}" if args.pred_by == 'target' else f'best_cls_{args.split_ratio}'
    else:
        ckpt_file_name = args.ckpt_file_name

    mode_dir = f'{args.mode}_no_in' if args.no_in else args.mode
    checkpoint_path = os.path.join(args.root_dir, args.sub_dir, mode_dir, f'{ckpt_file_name}.ckpt')
    test_data_dir = os.path.join(args.root_dir, args.sub_dir, mode_dir)
    print(checkpoint_path)

    de = DatasetEvaluation(args.model_name, checkpoint_path, args.mode, args.pred_by, args.concat_or_mean)
    de.evaluate(test_data_dir)

if __name__ == '__main__':
    main()