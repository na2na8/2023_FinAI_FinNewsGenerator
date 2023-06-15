import argparse
import numpy as np
import os
import torch
import random

from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from KeywordNewsDataLoader import *
from KeywordNewsGenerator import *

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword_model', type=str, default='/home/nykim/finAI/4_models/keyword_unique')
    parser.add_argument('--model', type=str, default='gogamza/kobart-base-v1')
    parser.add_argument('--tokenizer', type=str, default='gogamza/kobart-base-v1')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ckpt_model', type=str, default=None)
    parser.add_argument('--mode', type=str, default="both2", help="[both1, both2]")
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--ckpt_path', type=str, default='/home/nykim/finAI/0_numbers')
    parser.add_argument('--tensorboard', type=str, default='both2')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    set_random_seed(random_seed=args.randomseed)
    
    tokenizer.add_tokens(['[FILING]', '[KEYWORD]'], special_tokens=True)
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='VALID_ROUGE2',
                                                    dirpath=os.path.join(args.ckpt_path, args.tensorboard),
                                                    filename=str(args.randomseed) + f'-{args.learning_rate}' + '-{epoch:02d}-{VALID_ROUGE1:.3f}-{VALID_ROUGE2:.3f}-{VALID_ROUGEL:.3f}',
                                                    verbose=False,
                                                    save_last=True,
                                                    mode='max',
                                                    save_top_k=1,
                                                    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_path, 'lightning_logs'), name=args.tensorboard)
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    device = torch.device("cuda:1")
    
    dm = KeywordNewsDataLoader(tokenizer, args, device)
    trainer = pl.Trainer(
        default_root_dir= args.ckpt_path,
        logger = tb_logger,
        callbacks = [checkpoint_callback, lr_logger],
        max_epochs=args.epoch,
        accelerator='gpu',
        devices=[1],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2
    )
    model = KeywordNewsGenerator(device, args, tokenizer)

    trainer.fit(model, dm)