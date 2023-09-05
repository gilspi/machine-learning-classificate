import os
import shutil
import pathlib
import random

# base dir
BASE_DIR_ABSOLUTE = os.getcwd() # вощвращает текущую директорию
print(f'Текущая директория: {os.getcwd()}')
OUT_DIR_NAME = './dataset/flowers_prepared/'
SRC_DIR = './dataset/flowers/'

# out dir
TRAIN_OUT_DIR = OUT_DIR_NAME + 'train/'
VAL_OUT_DIR = OUT_DIR_NAME + 'test/'

# input dirs
input_dirs = [
    './dataset/flowers/daisy/',
    './dataset/flowers/dandelion/',
    './dataset/flowers/roses/',
    './dataset/flowers/sunflowers/',
    './dataset/flowers/tulips/',
]

# Preparing images data by rule 80/20
print('Preparing images data by rule 80/20.\n')

# создание пустой папки
def mkemptydir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'Создание директории успешно выполнено: {path}')
        print(f'Текущая директория: {os.getcwd()}')

# функция копирования определенного количества картинок в другую папку
def copy_imgs(src):
    dir_name = pathlib.Path(src).name
    output_test = f'./dataset/flowers_prepared/test/{dir_name}/'
    output_train = f'./dataset/flowers_prepared/train/{dir_name}/'

    imgs = list(pathlib.Path(src).glob('*.jpg'))  # список картинок из src 
    total_images = len(imgs) # общее число картинок из src
    percent_to_move = int(total_images * 0.2)  # 20% надо переместить для тестов
    moved_imgs = random.sample(imgs, percent_to_move)  # список картинок, которые нужно переместить
    
    # Удаляем все существующие файлы и папки
    if os.path.exists(output_test):
        shutil.rmtree(output_test)
    if os.path.exists(output_train):
        shutil.rmtree(output_train)


    # Создаем папки только один раз
    mkemptydir(output_test)
    mkemptydir(output_train)
    
    print(f'Имя папки: {dir_name}')
    print(f'Количество изображений: {len(imgs)}')

    for img in moved_imgs:
        shutil.copy(img, output_test)

    for img in imgs:
        if img not in moved_imgs:
            shutil.copy(img, output_train)

    print(f'Количество изображений в папке "Тест": {len(list(pathlib.Path(output_test).glob("*.jpg")))}')
    print(f'Количество изображений в папке "Обучение": {len(list(pathlib.Path(output_train).glob("*.jpg")))}')
    print()


mkemptydir(OUT_DIR_NAME)
mkemptydir(TRAIN_OUT_DIR)
mkemptydir(VAL_OUT_DIR)


for input_dir in input_dirs:
    copy_imgs(src=input_dir)
