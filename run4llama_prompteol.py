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
from transformers import AutoTokenizer, AutoConfig
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
from models.modeling_llama_lr import LlamaForCausalLM
from utils.tasks import ALL_TASKS
from utils.utils import move_to_cuda

def transform_func(examples, tokenizer, template, max_length):
    examples['input_texts'] = [template.format(text=text) for text in examples['input_texts']]
    batch_dict = tokenizer(examples['input_texts'], padding=True, truncation=False) #left padding
    length = len(batch_dict['input_ids'][0])
    if length > max_length: # need to truncate
        for i in range(len(batch_dict['input_ids'])): 
            if sum(batch_dict['attention_mask'][i]) > max_length:
                idx_start = length - sum(batch_dict['attention_mask'][i])
                batch_dict['input_ids'][i] = batch_dict['input_ids'][i][idx_start:idx_start+(max_length // 2)] + batch_dict['input_ids'][i][-max_length // 2:]
                batch_dict['attention_mask'][i] = batch_dict['attention_mask'][i][idx_start:idx_start+(max_length // 2)] + batch_dict['attention_mask'][i][-max_length // 2:]
            else:
                batch_dict['input_ids'][i] = batch_dict['input_ids'][i][-max_length:]
                batch_dict['attention_mask'][i] = batch_dict['attention_mask'][i][-max_length:]
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
        
    @torch.no_grad()
    def encode(self, sentences, l2_normalize=True, is_query=True, **kwargs) -> np.ndarray:
        dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(transform_func, tokenizer=self.tokenizer, template=self.template, max_length=self.max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size_per_gpu, shuffle=False, drop_last=False, num_workers=4, collate_fn=data_collator, pin_memory=True)
        
        encoded_embeds = []
        for batch in tqdm(data_loader):
            batch = move_to_cuda(batch)

            outputs = self.encoder(**batch, use_cache=False)
            embeds = outputs.last_hidden_state[:, -1, :] # last token pooling
            if l2_normalize:
                embeds = F.normalize(embeds, p=2, dim=-1)
            
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
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--filter_ratio", type=int, default=1)
    args = parser.parse_args()

    pretrained_model = os.path.join("/data/xxx/public", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, padding_side="left")
    tokenizer.pad_token_id = 0
    config = AutoConfig.from_pretrained(pretrained_model, _attn_implementation="flash_attention_2")
    model = LlamaForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16, config=config)

    model.set_lm_embed(config.hidden_size // args.filter_ratio)
    print(f"dim = {config.hidden_size // args.filter_ratio}")
    model = model.model

    template = "Summarize the sentence: \"{text}\" in one word:\""
    model = EncoderWrapper(model=model, tokenizer=tokenizer, template=template)

    for task_type, tasks in ALL_TASKS:
        model.encode = partial(model.encode, l2_normalize=False) if task_type in ["Classification"] else partial(model.encode, l2_normalize=True)
        
        evaluation = MTEB(tasks=tasks, task_langs=["en", "eng-Latn"])
        results = evaluation.run(
            model, 
            eval_splits=["test"], 
            output_folder=os.path.join("./output", f"{args.model_name}_prompteol_{args.filter_ratio}", task_type),
            verbosity=0,
            overwrite_results=False,
        )
    
    end_time = time.time()
    print(f"ALL_TASKS done! Total time: {(end_time - start_time) / 60} minutes")

# CUDA_VISIBLE_DEVICES=2 nohup python run4llama_prompteol.py --model_name llama-3.1-8b-instruct --filter_ratio 1 > llama1.log 2>&1 &