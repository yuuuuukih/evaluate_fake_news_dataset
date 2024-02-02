import os
import random
import numpy as np
import torch

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from finetune.model_pred_by_cls import BinaryClassifierByCLS
from finetune.model_pred_by_target import BinaryClassifierByTARGET
from finetune.data import FakeNewsDataModule

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--pred_by', default='target', choices=['cls', 'target'])
    parser.add_argument('--concat_or_mean', default='concat', choices=['concat', 'mean'])

    parser.add_argument('--mode', default='base', choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument("--root_dir", default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--file_name", default='default')
    parser.add_argument("--split_ratio", default='100', choices=['25', '50', '75', '100'])
    parser.add_argument('--no_in', default=False, action='store_true')

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--lr", default=2e-5, type=float, help='learning rate')
    parser.add_argument("--dropout_rate", default=0.1)

    parser.add_argument("--patience", default=2, type=int)

    args = parser.parse_args()

    # Set directory path.
    mode_dir = f'{args.mode}_no_in' if args.no_in else args.mode
    data_dir = os.path.join(args.root_dir, args.sub_dir, mode_dir)
    exp_dir = os.path.join(args.root_dir, args.sub_dir, mode_dir)

    if args.pred_by == 'target':
        model = BinaryClassifierByTARGET(model_name=args.model_name, lr=args.lr, dropout_rate=args.dropout_rate, batch_size=args.batch_size, concat_or_mean=args.concat_or_mean, save_transformer_model_path=None)
    elif args.pred_by == 'cls':
        model = BinaryClassifierByCLS(model_name=args.model_name, lr=args.lr, dropout_rate=args.dropout_rate, add_target_token=False if args.mode == 'base' else True, save_transformer_model_path=None)

    data_module = FakeNewsDataModule(
        data_dir=data_dir,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_seq_len,
        training_data_name=f'train_{args.split_ratio}'
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=True, mode='min', min_delta=0.0001)

    # logger = TensorBoardLogger('tb_logs', name='my_model')

    target_label_for_file_name: bool = args.pred_by == 'target'
    file_name = f"best_{args.pred_by}" + target_label_for_file_name * f"_{args.concat_or_mean}" + f"_{args.split_ratio}" if args.file_name == 'default' else args.file_name
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=exp_dir,
        filename=file_name,
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        # logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=args.max_epochs,
        deterministic=True, # for making seed fixed
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        # devices=[0, 1, 2, 3],
        default_root_dir=exp_dir
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()