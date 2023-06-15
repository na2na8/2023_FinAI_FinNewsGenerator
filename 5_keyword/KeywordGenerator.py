import re
import os

import torch
import torch.nn as nn

from rouge import Rouge
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_looggers
from transformers import BartForConditionalGeneration, AdamW

class KeywordGenerator(pl.LightningModule) :
    def __init__(self, device, args, tokenizer) :
        super().__init__()
        self._device = device
        
        self.learning_rate = args.learning_rate
        self.epoch = args.epoch
        self.tensorboard = args.tensorboard
        
        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(args.model)
        
        self.rouge = Rouge()
        
        self.preds = []
        self.trgts = []
        self.save_hyperparameters()
        
    def unique(self, predicts) :
        unique_predict = []
        for predict in predicts :
            predict = predict.split(' ')
            unique = []
            for idx in range(len(predict)) :
                word = predict[idx]
                if word not in unique :
                    unique.append(word)
            predict = ' '.join(unique)
            unique_predict.append(predict)
        return unique_predict
        
    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels
    ) :
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        return outputs
    
    def get_rouge(self, preds, targets) :
        scores = []
        for idx in range(len(preds)) :
            if not preds[idx] :
                scores.append(
                    {
                        'rouge-1' : {'f' : 0.0},
                        'rouge-2' : {'f' : 0.0},
                        'rouge-l' : {'f' : 0.0}
                    }
                )
            else :
                scores.append(self.rouge.get_scores(preds[idx], targets[idx])[0])
        # scores = self.rouge.get_scores(preds, targets)
        rouge1 = torch.mean(torch.tensor([score['rouge-1']['f'] for score in scores]))
        rouge2 = torch.mean(torch.tensor([score['rouge-2']['f'] for score in scores]))
        rougel = torch.mean(torch.tensor([score['rouge-l']['f'] for score in scores]))
        
        return rouge1, rouge2, rougel
    
    def training_step(self, batch, batch_idx, state='train') :
        outputs = self(
            input_ids=batch['input_ids'].to(self._device),
            attention_mask=batch['attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_attention_mask'].to(self._device),
            labels=batch['labels'].to(self._device)
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        preds = self.tokenizer.batch_decode(torch.argmax(logits, dim=2).cpu().detach(), skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(batch['decoder_input_ids'].cpu().detach(), skip_special_tokens=True)
        
        rouge1, rouge2, rougel = self.get_rouge(preds, targets)
        
        self.log(f"[{state.upper()} LOSS]", loss, prog_bar=True)
        self.log(f"[{state.upper()} ROUGE1]", rouge1, prog_bar=True)
        self.log(f"[{state.upper()} ROUGE2]", rouge2, prog_bar=True)
        self.log(f"[{state.upper()} ROUGEL]", rougel, prog_bar=True)
        
        return {
            'loss' : loss,
            'rouge1' : rouge1,
            'rouge2' : rouge2,
            'rougel' : rougel
        }
        
    def validation_step(self, batch, batch_idx, state='valid') :
        generated = self.model.generate(batch['input_ids'].to(self._device), max_length=512)
        
        targets = self.tokenizer.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        preds = self.unique(preds)
        
        self.preds += preds
        self.trgts += targets
        
        rouge1, rouge2, rougel = self.get_rouge(preds, targets)
        
        self.log(f"[{state.upper()} ROUGE1]", rouge1, prog_bar=True)
        self.log(f"[{state.upper()} ROUGE2]", rouge2, prog_bar=True)
        self.log(f"[{state.upper()} ROUGEL]", rougel, prog_bar=True)
        
        return {
            'rouge1' : rouge1,
            'rouge2' : rouge2,
            'rougel' : rougel
        }
        
    def training_epoch_end(self, outputs, state='train') : 
        loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        rouge1 = torch.mean(torch.tensor([output['rouge1'] for output in outputs]))
        rouge2 = torch.mean(torch.tensor([output['rouge2'] for output in outputs]))
        rougel = torch.mean(torch.tensor([output['rougel'] for output in outputs]))
        
        self.log(f'{state.upper()}_LOSS', loss, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_ROUGE1', rouge1, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_ROUGE2', rouge2, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_ROUGEL', rougel, on_epoch=True, prog_bar=True)
        
    def validation_epoch_end(self, outputs, state='valid') :
        rouge1 = torch.mean(torch.tensor([output['rouge1'] for output in outputs]))
        rouge2 = torch.mean(torch.tensor([output['rouge2'] for output in outputs]))
        rougel = torch.mean(torch.tensor([output['rougel'] for output in outputs]))
        
        self.log(f'{state.upper()}_ROUGE1', rouge1, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_ROUGE2', rouge2, on_epoch=True, prog_bar=True)
        self.log(f'{state.upper()}_ROUGEL', rougel, on_epoch=True, prog_bar=True)
        
        if self.current_epoch == 0 and not os.path.isfile(f'./keyword_outs/{self.tensorboard}_trgts.txt') :
            with open(f'./keywords_outs/{self.tensorboard}_trgts.txt', 'a') as f :
                str_trgts = '\n\n'.join(self.trgts)
                f.write(str_trgts)
                f.close()
            
        with open(f'./keyword_outs/{self.tensorboard}_preds_{self.current_epoch}.txt', 'a') as f :
            str_preds = '\n\n'.join(self.preds)
            f.write(str_preds)
            f.close()
        
        self.trgts.clear()
        self.preds.clear()
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]