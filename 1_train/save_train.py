import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
# from KeywordGenerator import *
from NewsGenerator import *
import argparse
import torch



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    g1 = parser.add_argument_group("CommonArgument")
    # g1.add_argument("--path", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/data/dedup/train")
    g1.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--model", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--ckpt_path", type=str, default=None)
    g1.add_argument("--model_path", type=str, default=None)
    g1.add_argument("--tensorboard", type=str, default="wo_comma")
    g1.add_argument('--mode', type=str, default="numbers2", help="[keyword, filing, both1, both2, numbers1, numbers2]")
    # g1.add_argument("--gpu_lists", type=str, default="0", help="string; make list by splitting by ','") # gpu list to be used

    g2 = parser.add_argument_group("TrainingArgument")
    # g2.add_argument("--ckpt_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/ckpt/test')
    g2.add_argument("--epoch", type=int, default=1)
    g2.add_argument("--learning_rate", type=float, default=1e-5)
    # g2.add_argument("--num_warmup_steps", type=int, default=10000)
    g2.add_argument("--max_length", type=int, default=512)
    g2.add_argument("--batch_size", type=int, default=16)

    # g2.add_argument("--min_learning_rate", type=float, default=1e-5)
    # g2.add_argument("--max_learning_rate", type=float, default=1e-4)
    g2.add_argument("--num_workers", type=int, default=8)
    # g2.add_argument("--logging_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/log/test')

    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if 'both' in args.mode :
        tokenizer.add_tokens(['[FILING]', '[KEYWORD]'], special_tokens=True)
    elif 'numbers' in args.mode :
        tokenizer.add_tokens(['[FILING]', '[NUMBERS]'], special_tokens=True)
    
    model_path = '/home/nykim/finAI/0_numbers/numbers2'

    ckpt_path = "/home/nykim/finAI/0_numbers/numbers2/42-1e-05-numbers2-epoch=09-VALID_ROUGE1=0.565-VALID_ROUGE2=0.404-VALID_ROUGEL=0.559.ckpt"
    ckpt_model = NewsGenerator.load_from_checkpoint(checkpoint_path=ckpt_path, args=args, device=device, tokenizer=tokenizer)
    ckpt_model.model.model.save_pretrained(model_path)