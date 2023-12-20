import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from finetune.model import BertBinaryClassifier
from finetune.data import FakeNewsDataModule

def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', default='base', choices=['base', 'pre_target_timeline', 'all_timeline'])
    parser.add_argument("--root_dir", default='/mnt/mint/hara/datasets/news_category_dataset/dataset')
    parser.add_argument('--sub_dir', default='', help='e.g., diff7_rep1, diff7_rep3, diff7_ins1, diff6_rep1, diff6_rep3, diff6_ins1')
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--file_name", default='best')

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--dropout_rate", default=0.1)

    parser.add_argument("--patience", default=2, type=int)

    args = parser.parse_args()

    # Set directory path.
    data_dir = os.path.join(args.root_dir, args.sub_dir, args.mode)
    exp_dir = os.path.join(args.root_dir, args.sub_dir, args.mode)

    model = BertBinaryClassifier(model_name=args.model_name, lr=args.lr, dropout_rate=args.dropout_rate, add_target_token=False if args.mode == 'base' else True, save_transformer_model_path=None)
    data_module = FakeNewsDataModule(
        data_dir=data_dir,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_seq_len
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=True, mode='min', min_delta=0.0001)

    # logger = TensorBoardLogger('tb_logs', name='my_model')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=exp_dir,
        filename=args.file_name,
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        # logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=args.max_epochs,
        deterministic=True,
        # strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        # devices=[0, 1, 2, 3],
        default_root_dir=exp_dir
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()