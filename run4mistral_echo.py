import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 在导入 numpy 前设置！
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from functools import partial
from datasets import Dataset
from mteb import MTEB
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoConfig, AutoModel

from models.modeling_mistral_lr import MistralForCausalLM
from utils.tasks import ALL_TASKS
from utils.utils import move_to_cuda, get_task_def_by_task_name_and_type

def transform_func(examples, tokenizer, template, max_length):
    template_pieces = [part for part in template.split("{text}") if part]
    tokenized_template_pieces = [tokenizer(piece, padding=False, truncation=False) for piece in template_pieces]

    batch_dict = tokenizer(examples['input_texts'], padding=False, truncation=True, max_length=max_length)
    batch_dict['embed_mask'] = []
    for i in range(len(batch_dict['input_ids'])):
        batch_dict['embed_mask'].append([0] + [0] * len(tokenized_template_pieces[0]['input_ids']) + [0] * len(batch_dict['input_ids'][i]) + [0] * len(tokenized_template_pieces[1]['input_ids']) + [1] * len(batch_dict['input_ids'][i]) + [1])
        batch_dict['input_ids'][i] = [tokenizer.bos_token_id] + tokenized_template_pieces[0]['input_ids'] + batch_dict['input_ids'][i] + tokenized_template_pieces[1]['input_ids'] + batch_dict['input_ids'][i] + [tokenizer.eos_token_id]
        batch_dict['attention_mask'][i] = [1] + tokenized_template_pieces[0]['attention_mask'] + batch_dict['attention_mask'][i] + tokenized_template_pieces[1]['attention_mask'] + batch_dict['attention_mask'][i] + [1]

    max_tokenized_length = max(len(_) for _ in batch_dict['input_ids'])
    for i in range(len(batch_dict['input_ids'])): #padding
        pad_length = max_tokenized_length - len(batch_dict['input_ids'][i])
        if pad_length > 0:
            if tokenizer.padding_side == 'left':
                batch_dict['input_ids'][i] = [tokenizer.pad_token_id] * pad_length + batch_dict['input_ids'][i]
                batch_dict['attention_mask'][i] = [0] * pad_length + batch_dict['attention_mask'][i]
                batch_dict['embed_mask'][i] = [0] * pad_length + batch_dict['embed_mask'][i]
            elif tokenizer.padding_side == 'right':
                batch_dict['input_ids'][i] = batch_dict['input_ids'][i] + [tokenizer.pad_token_id] * pad_length
                batch_dict['attention_mask'][i] = batch_dict['attention_mask'][i] + [0] * pad_length
                batch_dict['embed_mask'][i] = batch_dict['embed_mask'][i] + [0] * pad_length
    return batch_dict

class EncoderWrapper:
    def __init__(self, model, tokenizer, template="{text}"):
        self.encoder = model
        self.tokenizer = tokenizer
        self.template = template
        self.gpu_num = torch.cuda.device_count()
        self.batch_size_per_gpu = 64
        self.max_length = 512

        self.encdoer = self.encoder.cuda()
        self.encoder.eval()

        if self.gpu_num > 1:
            self.encoder = nn.DataParallel(self.encoder)

        self.instruct = ""
        self.templates = {
            'query': 'Rewrite the following paragraph: {text}. The rewritten paragraph: {text}',
            'document': 'Rewrite the following paragraph: {text}. The rewritten paragraph: {text}',
        }
        
    @torch.no_grad()
    def encode(self, sentences, l2_normalize=True, is_query=True, **kwargs) -> np.ndarray:
        if is_query:
            self.template = (self.templates['query']).format(instruct=self.instruct, text="{text}")
        else:
            self.template = (self.templates['document'])
        dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(transform_func, tokenizer=self.tokenizer, template=self.template, max_length=self.max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size_per_gpu, shuffle=False, drop_last=False, num_workers=32, collate_fn=data_collator, pin_memory=True)
        
        encoded_embeds = []
        for batch in tqdm(data_loader):
            batch = move_to_cuda(batch)

            outputs = self.encoder(**batch, use_cache=False)

            embeds = torch.sum(outputs.last_hidden_state.to(dtype=torch.float64) * batch['embed_mask'].unsqueeze(-1), dim=1) / torch.sum(batch['embed_mask'], dim=1).unsqueeze(-1) # mean pooling
            embeds.masked_fill_(torch.isnan(embeds), 0)

            encoded_embeds.append(embeds.to(dtype=torch.float64).detach().cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    @torch.no_grad()
    def encode_queries(self, sentences, **kwargs) -> np.ndarray:
        return self.encode(sentences, is_query=True, **kwargs)

    @torch.no_grad()
    def encode_corpus(self, sentences, **kwargs) -> np.ndarray:
        if isinstance(sentences[0], dict):
            sentences = [
                (doc["title"] + " " + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in sentences
            ]
        return self.encode(sentences, is_query=False, **kwargs)


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Mistral-7B-Instruct-v0.3")
    parser.add_argument("--filter_ratio", type=int, default=1)
    args = parser.parse_args()

    pretrained_model = os.path.join("/data/xxx/public", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, padding_side="left", add_bos_token=False, add_eos_token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(pretrained_model, _attn_implementation="flash_attention_2")
    model = MistralForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16, config=config)

    model.set_lm_embed(config.hidden_size // args.filter_ratio)
    model = model.model

    template = "Summarize the sentence: \"{text}\" in one word:\""
    model = EncoderWrapper(model=model, tokenizer=tokenizer, template=template)

    for task_type, tasks in ALL_TASKS:
        model.encode = partial(model.encode, l2_normalize=False) if task_type in ["Classification"] else partial(model.encode, l2_normalize=True)
        
        for task_name in tasks:
            instruct = get_task_def_by_task_name_and_type(task_name, task_type)
            model.instruct = instruct

            evaluation = MTEB(tasks=[task_name], task_langs=["en", "eng-Latn"])
            results = evaluation.run(
                model, 
                eval_splits=["test"], 
                output_folder=os.path.join("./output", f"{args.model_name}_echo_{args.filter_ratio}", task_type),
                verbosity=0,
                overwrite_results=False,
            )
    
    end_time = time.time()
    print(f"ALL_TASKS done! Total time: {(end_time - start_time) / 60} minutes")

# CUDA_VISIBLE_DEVICES=4 nohup python run4mistral_echo.py --model_name Mistral-7B-Instruct-v0.3 --filter_ratio 1 > mistral11.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python run4mistral_echo.py --model_name Mistral-7B-Instruct-v0.3 --filter_ratio 2 > mistral22.log 2>&1 &