import json
import pandas as pd
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class FinDataset(Dataset) :
    def __init__(self, path, tokenizer, args) :
        self.data = self.get_dataframe(path)
        self.args = args
        self.tokenizer = tokenizer
        
    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, idx) :
        # <s> + sentence
        if self.args.mode == 'keyword' :
            sentence = self.tokenizer.bos_token + self.data['keyword'].iloc[idx]
        elif self.args.mode == 'filing' :
            sentence = self.tokenizer.bos_token + self.data['filing_content'].iloc[idx]
        elif self.args.mode == 'both1' :
            sentence = self.tokenizer.bos_token + '[KEYWORD]' + self.data['keyword'].iloc[idx] + '[FILING]' + self.data['filing_content'].iloc[idx]
        elif self.args.mode == 'both2' :
            sentence = self.tokenizer.bos_token + '[FILING]' + self.data['filing_content'].iloc[idx] + '[KEYWORD]' + self.data['keyword'].iloc[idx]
        elif self.args.mode == 'numbers1' :
            sentence = self.tokenizer.bos_token + '[NUMBERS]' + self.numbers_preprocess(self.data['filing_content'].iloc[idx]) + '[FILING]' + self.data['filing_content'].iloc[idx]
        elif self.args.mode == 'numbers2' :
            sentence = self.tokenizer.bos_token + '[FILING]' + self.data['filing_content'].iloc[idx] + '[NUMBERS]' + self.numbers_preprocess(self.data['filing_content'].iloc[idx])
            
            
        article_content = self.data['article_content'].iloc[idx]
        
        decoder_input = self.tokenizer.eos_token + self.tokenizer.bos_token + article_content
        label_input = self.tokenizer.bos_token + article_content + self.tokenizer.eos_token 
        
        encoder_inputs = self.tokenizer(
            sentence, return_tensors='pt', add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True
        )
        decoder_inputs = self.tokenizer(
            decoder_input, return_tensors='pt', add_special_tokens=True, max_length=self.args.max_length, padding='max_length', truncation=True
        )
        tok_label = self.tokenizer(label_input)['input_ids']
        label_inputs = torch.tensor(tok_label + [-100] * (self.args.max_length - len(tok_label)))
        
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
                # article_content = data[idx]['article_content']
                article_content = self.article_preprocess(data[idx]['article_content'])
                filing_content = self.filing_preprocess(data[idx]['filing_content'])
                keyword = self.keyword_preprocess(data[idx]['article']['keyword'])
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
                    keyword = self.keyword_preprocess(data[idx]['article']['keyword'])
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
        numbers = re.findall(r'[0-9\,\-\.]+', sentence)
        numbers = list(filter(lambda x : x != '.' and x != '-', numbers))
        return ' '.join(numbers)

    def keyword_preprocess(self, keyword) :
        # key_list = keyword.split(',')
        # random.shuffle(key_list)
        # keyword = ' '.join(key_list)
        
        # keyword = keyword.replace(',', ' ')
        
        keyword = keyword.split(',')
        unique = []
        for idx in range(len(keyword)) :
            word = keyword[idx]
            if word not in unique :
                unique.append(word)
        keyword = ' '.join(unique)
        return keyword
    
class FinDataLoader(pl.LightningDataModule) :
    def __init__(self, tokenizer, args) :
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        
        self.setup()
        
    def setup(self, stage=None) :
        self.set_train = FinDataset(
            '/home/nykim/finAI/3_data_publish/train/meta.json',
            self.tokenizer,
            self.args
        )
        
        self.set_valid = FinDataset(
            '/home/nykim/finAI/3_data_publish/valid/meta.json',
            self.tokenizer,
            self.args
        )
        
        # self.set_test = FinDataset(
        #     '/home/nykim/finAI/3_data_publish/test/one_sample/meta1.json',
        #     self.tokenizer,
        #     self.args
        # )
        
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
