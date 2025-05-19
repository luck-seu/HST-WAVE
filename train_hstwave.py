from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.loader import DataLoader
from model import HSTWAVE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import numpy as np
import gc


pl.seed_everything(295)
parser = ArgumentParser()
parser.add_argument('--data', type=str, default='jh', help="dataset.")
parser.add_argument('--seq_len', type=int, default=12, help="Time input length.")
parser.add_argument('--horizen', type=int, default=12, help="Time output length.")
parser.add_argument('--in_channels', type=int, default=3, help="The dimension of inputs.")
parser.add_argument('--out_channels', type=int, default=1, help="The dimension of outputs.")
# hyper parameters
parser.add_argument('--hidden_dim', type=int, default=32, help="The hidden dimension of models.")
parser.add_argument('--temporal_dim', type=int, default=32, help="Time embedding dimension.")
parser.add_argument('--align_dim', type=int, default=512, help="The align dimension of models.")
parser.add_argument('--num_heads', type=int, default=4, help='The num heads of attention.')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout prob.')
parser.add_argument('--num_layers', type=int, default=2, help="Num of Transformer encoder layers.")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
parser.add_argument('--clip_val', type=float, default=5, help="Gradient clipping values. ")
parser.add_argument('--total_epochs', type=int, default=100, help="Max epochs of model training.")
parser.add_argument('--lr_decay_step', type=int, default=100, help="Learning rate decay step size.")
parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
parser.add_argument('--gpu_lst', nargs='+', type=int, help="Which gpu to use.")
parser.add_argument('--accumulate_grad_batches', type=int, default=8, help="Gradient accumulation steps.")
args = parser.parse_args()

def train():

    # load data
    train_dataset, hw_scaler, _ = load_dataset("train")
    train_dataset = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataset, _, _ = load_dataset("val")
    valid_dataset = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset, _, _ = load_dataset("test")
    test_dataset = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # load model
    pre_HSTG = HSTWAVE(in_channels=[[3],[64,64,64],[args.seq_len]], trainmode='base', 
                    batch=args.batch_size, seq_len=args.seq_len, horizen=args.horizen, 
                    scaler=hw_scaler, num_nodes=[130,13], metadata=get_metadata(), lr=args.lr,
                    weight_decay=args.weight_decay, lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma, is_large_label=True)
    print("agrs: ", args)

    checkpoint_callback = ModelCheckpoint(monitor="validation_epoch_average",
                                          filename='res' + '-{epoch:03d}-{validation_epoch_average:.4f}',
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True)

    # early stop
    early_stop_callback = EarlyStopping(
        monitor='validation_epoch_average',
        min_delta=0,  
        patience=50, 
        verbose=True, 
        mode='min'
        )

    trainer = pl.Trainer(callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), early_stop_callback],
                        gradient_clip_val=args.clip_val, 
                        max_epochs=args.total_epochs, 
                        devices=[0], 
                        accelerator='gpu',
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        log_every_n_steps=1
                        )
    trainer.fit(pre_HSTG, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)
    res = trainer.test(pre_HSTG, test_dataset, ckpt_path='best')
    print(res)
    # adapter model
    # adapter_HSTWAVE = AdapterHSTWAVE(pre_HSTG, in_channels=[[3],[64,64,64],[args.seq_len]], trainmode='adapter', batch=args.batch_size, seq_len=args.seq_len, horizen=args.horizen, scaler=hw_scaler, num_nodes=[130,13], metadata=get_metadata(), lr=args.lr,
    #                    weight_decay=args.weight_decay, lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma, is_large_label=True)
    # trainer.fit(adapter_HSTWAVE, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)

if __name__ == "__main__":
    if args.data == 'hz':
        from data.HWDataset import get_metadata, load_dataset
    elif args.data == 'jh':
        from data.JHDataset import get_metadata, load_dataset
    train()