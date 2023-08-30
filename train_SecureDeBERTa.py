# https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/language_modeling.ipynb#scrollTo=FX3tVGtTXKMp

import torch
import logging
# from Python_projects.TextGeneration.lib.CVE2CoA_functions import func_savejson
from functions import group_texts, read2list
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import DebertaTokenizer, DebertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tokenizers.implementations import ByteLevelBPETokenizer
import random
import os

def func_savejson(DICT, fname):
    import json
    with open(fname, 'w', encoding='iso-8859-1') as fout:
        json.dump(DICT, fout)
try:
    # get index of currently selected device
    logging.warning(torch.cuda.current_device()) # returns 0 in my case


    # get number of GPUs available
    logging.warning(torch.cuda.device_count()) # returns 1 in my case


    # get the name of the device
    logging.warning(torch.cuda.get_device_name(0)) 
except:
    pass

q="s"

class Config:
    if q=="s":
        model_name = "microsoft/deberta-v3-base"
        batch = 16
    else:
        model_name = "microsoft/deberta-v3-large"
        batch = 6
    tokenizer = "/users/eaghaei/Python_projects/TextGeneration/models/SecureDeBERTaTokenzier"
    # tokenizer = "microsoft/deberta-base"
    train_data = "/users/eaghaei/Python_projects/TextGeneration/data/SecureBERT_Dataset_2023.txt"
    # train_data = "/users/eaghaei/Python_projects/TextGeneration/dataset_512_uncased_m.txt"
    epochs = 5
    shuffle = False
    train_tokenizer = False
    

if Config.shuffle:
    fname = Config.train_data
    logging.warning("Data shuffle is ON!\nReading the dataset to a list")
    with open(Config.train_data, "r") as f:
        data = f.readlines()
    data = [d for d in data if d not in [""," ","\n"]]

    logging.warning("Shuffling data")
    random.shuffle(data)
    logging.warning(f"Saving new data to {fname}")
    with open(fname, "w") as f:
        f.writelines(data)
    Config.train_data = fname
    del data


if Config.train_tokenizer or not os.listdir(Config.tokenizer):
    logging.warning("Training new tokenizer")
#  tokenizer = GPT2TokenizerFast.from_pretrained(Config.tokenizer)
    tokenizer = ByteLevelBPETokenizer()
    # Customize training
    data_chunk_addresses = [Config.train_data]
    tokenizer.train(files=data_chunk_addresses, vocab_size=50257, show_progress=True, min_frequency=8,
                    special_tokens=["<s>",
                                    "<pad>",
                                    "</s>",
                                    "<sep>",
                                    "<unk>",
                                    "<mask>",
                                    "<bos>",
                                    "<eos>"
                                    ])
    logging.warning("Saving Tokenizer")
    tokenizer.save_model(Config.tokenizer)
    logging.warning("Tokenzier Saved!")
    del tokenizer
else:
    logging.warning("Chose to not train new tokenizer")

import torch
from torch.utils.data import Dataset
from transformers import  DataCollatorForLanguageModeling
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.file = open(filepath, 'r', encoding='utf-8')

    def __len__(self):
        self.file.seek(0)
        lines = 0
        buf_size = 1024 * 1024
        read_f = self.file.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)
        self.size = lines
        return self.size

    def __getitem__(self, i):
        line = self.file.readline()
        if not line:
            self.file.seek(0)
            line = self.file.readline()
        tokenized_text = self.tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=self.block_size)
        return {'input_ids': torch.tensor(tokenized_text)}
    

filepath = Config.train_data
logging.warning(f"Load Tokenizer")
tokenizer = DebertaTokenizer.from_pretrained(Config.tokenizer)
tokenizer.pad_token = tokenizer.eos_token
block_size = 512
dataset = TextDataset(filepath, tokenizer, block_size)


logging.warning(f"Load Model")

# model = DebertaForMaskedLM.from_pretrained(Config.model_name, ignore_mismatched_sizes=True)
model = DebertaForMaskedLM.from_pretrained("/users/eaghaei/Python_projects/TextGeneration/models/microsoft/deberta-v3-base/checkpoint-265000", ignore_mismatched_sizes=True)

batch_size = Config.batch
epochs = Config.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.warning(f"Model is running on {device}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True
)
training_args = TrainingArguments(
    output_dir=f"/users/eaghaei/Python_projects/TextGeneration/models/{Config.model_name}",
    overwrite_output_dir=True,
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=Config.batch,
    weight_decay=0.01,
    logging_dir='./logs-deberta',
    save_steps=7500,
    save_total_limit=2,
    logging_steps=7500,
    learning_rate=4e-5,
    adam_beta2=0.98,
    adam_epsilon=1e-08,
    warmup_steps=15000,
    load_best_model_at_end=True,
    resume_from_checkpoint=True

    # logging_strategy='steps',
    # save_strategy='steps',
    # fp16=True,
    # warmup_steps=1000,

)

# create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
    # callbacks=[print_training_loss]
)

# train model
trainer.train()