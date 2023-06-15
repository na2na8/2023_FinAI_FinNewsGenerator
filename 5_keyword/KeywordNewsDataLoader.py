import json
import pandas as pd
import random
import re
import os 

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration
import pytorch_lightning as pl

class KeywordNewsDataset(Dataset) :
    def __init__(self, path, tokenizer, args, device, stage) :
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        
        self._device = device
        
        self.keyword_model = BartForConditionalGeneration.from_pretrained(args.keyword_model).to(device)
        
        if not os.path.isfile(os.path.join('/home/nykim/finAI/3_data_publish', f'{stage}.csv')) :
            print("generate dataframe...")
            self.data = self.get_dataframe(path)
            self.data.to_csv(os.path.join('/home/nykim/finAI/3_data_publish', f'{stage}.csv'))
        else :
            self.data = pd.read_csv(os.path.join('/home/nykim/finAI/3_data_publish', f'{stage}.csv'))
        self.args = args
        
    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, idx) :
        # encoder inputs
        # get KEYWORD
        # <s> + filing
        # filing = self.tokenizer.bos_token + self.data['filing_content'].iloc[idx]
        # tokked_filing = self.tokenizer(
        #     filing, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        # )['input_ids']
        # tokked_keyword = self.keyword_model.generate(tokked_filing.to(self._device), max_length=512)[0]
        # keyword = self.keyword_preprocess(self.tokenizer.decode(tokked_keyword, skip_special_tokens=True))
        # tokked_keyword = None
        
        keyword = self.data['keyword'].iloc[idx]
        if self.args.mode == 'both1' :
            sentence = self.tokenizer.bos_token + '[KEYWORD]' + keyword + '[FILING]' + self.data['filing_content'].iloc[idx]
        elif self.args.mode == 'both2' :
            sentence = self.tokenizer.bos_token + '[FILING]' + self.data['filing_content'].iloc[idx] + '[KEYWORD]' + keyword
        
        # decoder inputs
        article_content = self.data['article_content'].iloc[idx]
        decoder_input = self.tokenizer.eos_token + self.tokenizer.bos_token + article_content
        label_input = self.tokenizer.bos_token + article_content + self.tokenizer.eos_token 
        
        encoder_inputs = self.tokenizer(
            sentence, return_tensors='pt', add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True
        )
        decoder_inputs = self.tokenizer(
            decoder_input, return_tensors='pt', add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True
        )
        tok_label = self.tokenizer(label_input)['input_ids'][:512]
        label_inputs = torch.tensor(tok_label + [-100] * (self.args.max_length - len(tok_label)))[:512]
        
        return {
            'input_ids' : encoder_inputs['input_ids'][0],
            'attention_mask' : encoder_inputs['attention_mask'][0],
            'decoder_input_ids' : decoder_inputs['input_ids'][0],
            'decoder_attention_mask' : decoder_inputs['attention_mask'][0],
            'labels' : label_inputs
        }
              
    def get_dataframe(self, path) :
        with open(path, 'r') as j :
            data = json.load(j)
            
        df = {
            'article_content' : [],
            'filing_content' : [],
            'keyword' : [],
            'title' : [],
            'dart_name' : [],
            'organizations' : []
        }
        
        for idx in range(len(data)) :
            detail_type = data[idx]['filing']['detail_type_name']
            if detail_type in ['수시공시', '공정공시', '주요사항보고서'] :
                article_content = self.article_preprocess(data[idx]['article_content'])
                filing_content = self.filing_preprocess(data[idx]['filing_content'])
                
                filing = self.tokenizer.bos_token + filing_content
                tokked_filing = self.tokenizer(
                    filing, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
                )['input_ids']
                tokked_keyword = self.keyword_model.generate(tokked_filing.to(self._device), max_length=512)[0]
                keyword = self.keyword_preprocess(self.tokenizer.decode(tokked_keyword, skip_special_tokens=True))
                
                # keyword = self.keyword_preprocess(data[idx]['article']['keyword'])
                title = data[idx]['article']['title']
                dart_name = data[idx]['company']['dart_name']
                organizations = data[idx]['article']['organizations']
                
                df['article_content'].append(article_content)
                df['filing_content'].append(filing_content)
                df['keyword'].append(keyword)
                df['title'].append(title)
                df['dart_name'].append(dart_name)
                df['organizations'].append(organizations)
            else :
                if 'valid' in path :
                    article_content = self.article_preprocess(data[idx]['article_content'])
                    filing_content = self.filing_preprocess(data[idx]['filing_content'])
                    
                    filing = self.tokenizer.bos_token + filing_content
                    tokked_filing = self.tokenizer(
                        filing, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
                    )['input_ids']
                    tokked_keyword = self.keyword_model.generate(tokked_filing.to(self._device), max_length=512)[0]
                    keyword = self.keyword_preprocess(self.tokenizer.decode(tokked_keyword, skip_special_tokens=True))
                    
                    title = data[idx]['article']['title']
                    dart_name = data[idx]['company']['dart_name']
                    organizations = data[idx]['article']['organizations']
                    
                    df['article_content'].append(article_content)
                    df['filing_content'].append(filing_content)
                    df['keyword'].append(keyword)
                    df['title'].append(title)
                    df['dart_name'].append(dart_name)
                    df['organizations'].append(organizations)
            
        df = pd.DataFrame(df)
        df = df.dropna(axis=0)
        
        return df
        
    def article_preprocess(self, article) :
        article = re.sub(r'[^\w,.%\(\)\-&]', ' ', article)
        return article
    
    def filing_preprocess(self, sentence) :
        sentence = sentence.replace('\n', ' ')
        return sentence
    
    def numbers_preprocess(self, sentence) :
        # 1. , 2. , ... 와 같은 숫자 . 띄어쓰기 제거
        sentence = sentence.replace('- ', '')
        sentence = re.sub(r'\d+\.\s+', '', sentence)
        sentence = re.sub(r'-(\w+)', r'\1', sentence)
        numbers =  ' '.join(re.findall(r'[0-9\,\-\.]+', numbers))
        return numbers
    
    def keyword_preprocess(self, keyword) :
        # key_list = keyword.split(',')
        # random.shuffle(key_list)
        # keyword = ' '.join(key_list)
        
        # keyword = keyword.replace(',', ' ')
        
        keyword = keyword.split(' ')
        unique = []
        for idx in range(len(keyword)) :
            word = keyword[idx]
            if word not in unique :
                unique.append(word)
        keyword = ' '.join(unique)
        return keyword
    
class KeywordNewsDataLoader(pl.LightningDataModule) :
    def __init__(self, tokenizer, args, device) :
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        
        self._device = device
        
        self.setup()
        
    def setup(self, stage=None) :
        self.set_train = KeywordNewsDataset(
            '/home/nykim/finAI/3_data_publish/train/meta.json',
            self.tokenizer,
            self.args,
            self._device,
            'train'
        )
        
        self.set_valid = KeywordNewsDataset(
            '/home/nykim/finAI/3_data_publish/valid/meta.json',
            self.tokenizer,
            self.args,
            self._device,
            'valid'
        )
        
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
