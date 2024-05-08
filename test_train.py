#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
#import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification


#   Подключаем локальные модели
model = GPT2LMHeadModel.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2', local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2', local_files_only=True)
#   Открываем текст для тюнинга модели
fil = open('1.txt', 'r', encoding='utf-8')
text = fil.read()
#   Сохраним обучающие данные в .txt файл
train_path = 'train_dataset.txt'
with open(train_path, "w", encoding='utf-8') as f:
    f.write(text)
#   Создание датасета
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)

#   Создание даталодера (нарезает текст на оптимальные по длине куски)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="models", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output dir
    num_train_epochs=200, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=10, # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=16, # to make "virtual" batch size larger
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), None)
)
trainer.train()
