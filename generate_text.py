#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Если torch установился без поддержки cuda, то выполняем следующие команды
pip uninstall torch
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
"""
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer  #, AutoTokenizer, AutoModelForSequenceClassification
import warnings

'''Отключаем предупреждения, т.к. в одной из моих библиотек какая-то срань про устаревшую процедуру и скорую замену'''
warnings.filterwarnings('ignore')

class GenerateText:
    def __init__(self):
        self.text = ''
        self.args = None
        self.text = ''
        self.local_model_path = 'ai-forever/rugpt3medium_based_on_gpt2'
        self.local_file_only = True
        self.model = None
        self.tokenizer = None
        self.inpt = None
        self.out = None
        self.generated_text = ''
        self.con = False
        self.file = None
        self.max_len = 300

    '''Обработка командной строки return: string'''
    def argument_processing(self):
        parser = argparse.ArgumentParser(description="Пример использования argparse")
        parser.add_argument("-s", "--text", nargs='+', help="Введите текст для затравки", type=str,
                            required=True)
        parser.add_argument("-f", "--file", help="Введите имя файла для сохранения результата",
                            type=str, required=False)
        parser.add_argument("-с", "--con", help="Выводить в консоль? (bool)", type=bool, required=False,
                            default=True)
        parser.add_argument("-m", "--max_len", help="Введите размер текста (int)", type=int, required=False,
                            default=300)

        arg = parser.parse_args()
        self.con = arg.con
        self.file = arg.file
        self.max_len = arg.max_len

        for i in arg.text:
            self.text = str(self.text) + ' ' + str(i)
        self.text = self.text[1:]

        return self.text

    '''Проверка на возможность использовать GPU return: Bool'''
    @staticmethod
    def check_cuda():
        ver = torch.version.cuda
        cuda_is_ok = torch.cuda.is_available()
        try:
            x = torch.Tensor([1, 2, 3]).cuda()
        except:
            print('Error CUDA')
            return False
        print(f"CUDA Enabled: {cuda_is_ok} / version: {ver} / CUDA test work: {x.device}")
        return cuda_is_ok

    '''Подключаем локальные модели'''
    def transformers_init(self):
        #   Подключаем локальные модели
        self.model = GPT2LMHeadModel.from_pretrained(self.local_model_path, local_files_only=self.local_file_only)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.local_model_path, local_files_only=self.local_file_only)

    '''Генерируем случайный текст'''
    def Generate(self):
        #   Обрабатываем аргументы.
        #   Задаем желаемое начала текста, то есть затравку.
        self.argument_processing()
        #   Подключаем локальные модели
        self.transformers_init()
        #   Переведем работу модели на GPU если возможно
        self.model.cuda()
        #   Закодируем затравку на "язык" модели
        self.inpt = self.tokenizer.encode(self.text, return_tensors="pt")
        #   Получаем сгенерированный кодированный список
        self.out = self.model.generate(self.inpt.cuda(), max_length=300, repetition_penalty=6.0, do_sample=True,
                                       top_k=5, top_p=0.95, temperature=1, no_repeat_ngram_size=2)
        #   Декодируем в текст
        self.generated_text = list(map(self.tokenizer.decode, self.out))[0]
        return self.generated_text

    '''Удаляем класс'''
    def __del__(self): return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''Создаем экземпляр класса'''
    Gena = GenerateText()
    '''Проверяем CUDA хотя нах не нужно'''
    Gena.check_cuda()
    '''Генерируем текст'''
    generated_text = Gena.Generate()
    if Gena.file:
        '''Пишем текст в файл'''
        try:
            out_file = open(Gena.file, 'w', encoding='utf-8')
            out_file.write(generated_text)
        except IOError as e:
            print('Error: ', e)
    if Gena.con:
        '''Выводим текст в консоль'''
        print(generated_text)
