#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def check_cuda():
    print(torch.version.cuda)
    cuda_is_ok = torch.cuda.is_available()
    print(f"CUDA Enabled: {cuda_is_ok}")
    x = torch.Tensor([1, 2, 3])
    print(x.device)


#   Задаем желаемое начала текста, то есть затравку
text = "служебная записка касательно нарушений техники безопасности"

check_cuda()
#   Сделаем генерацию воспроизводимой
np.random.seed(42)
torch.manual_seed(42)
#   Подключаем локальные модели
model = GPT2LMHeadModel.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2', local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2', local_files_only=True)
#   Переведем работу модели на GPU
model.cuda()
#   Закодируем затравку на "язык" модели
inpt = tokenizer.encode(text, return_tensors="pt")
#   Получаем сгенерированный кодированный список
out = model.generate(inpt.cuda(), max_length=300, repetition_penalty=6.0, do_sample=True, top_k=5, top_p=0.95,
                     temperature=1, no_repeat_ngram_size=2)
#   Декодируем в текст
generated_text = list(map(tokenizer.decode, out))[0]
out_file = open('test_generation.txt', 'w', encoding='utf-8')
out_file.write(generated_text)
print(generated_text)
