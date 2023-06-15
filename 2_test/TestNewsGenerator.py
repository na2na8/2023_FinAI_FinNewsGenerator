import re
import os
import json

import torch
import torch.nn as nn
# from torchmetrics.text.rouge import ROUGEScore
from rouge import Rouge
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_looggers
from transformers import BartForConditionalGeneration, AdamW

class NewsGenerator(pl.LightningModule) :
    def __init__(self, device, args, tokenizer) :
        super().__init__()
        self._device = device
        
        self.learning_rate = args.learning_rate
        self.epoch = args.epoch
        self.tensorboard = args.tensorboard
        
        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(args.model)
        if 'both' in args.mode or 'numbers' in args.mode :
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # self.rouge = Rouge()
        
        self.json_dict = []

        self.save_hyperparameters()
        
    # def forward(
    #     self,
    #     input_ids,
    #     attention_mask,
    # ) :
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #     )
        
    #     return outputs
    
    # def get_rouge(self, preds, targets) :
    #     scores = []
    #     for idx in range(len(preds)) :
    #         if not preds[idx] :
    #             scores.append(
    #                 {
    #                     'rouge-1' : {'f' : 0.0},
    #                     'rouge-2' : {'f' : 0.0},
    #                     'rouge-l' : {'f' : 0.0}
    #                 }
    #             )
    #         else :
    #             scores.append(self.rouge.get_scores(preds[idx], targets[idx])[0])
    #     # scores = self.rouge.get_scores(preds, targets)
    #     rouge1 = torch.mean(torch.tensor([score['rouge-1']['f'] for score in scores]))
    #     rouge2 = torch.mean(torch.tensor([score['rouge-2']['f'] for score in scores]))
    #     rougel = torch.mean(torch.tensor([score['rouge-l']['f'] for score in scores]))
        
    #     return rouge1, rouge2, rougel
    
    def test_step(self, batch, batch_idx, state='test') :
        generated = self.model.generate(batch['input_ids'].to(self._device), max_length=512)
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        for idx in range(len(batch['input_ids'])) :
            j = {}
            j['filing_id'] = int(batch['filing_id'][idx])
            j['article_content'] = preds[idx]
            
            self.json_dict.append(j)
        
    def test_epoch_end(self, outputs, state='test') :
        with open('/home/nykim/finAI/2_test/outputs.json', 'w', encoding='utf8') as f :
            json.dump(self.json_dict, f, indent=4)
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]
        
        
        
        
        
            
