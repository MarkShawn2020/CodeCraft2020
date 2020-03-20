# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/3/20 6:48
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os
import re


ENCODING = "utf-8"

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
PACKAGE_NAME = os.path.basename(PROJECT_DIR)

RELEASE_DIR = os.path.join(PROJECT_DIR, "release")
os.makedirs(RELEASE_DIR, exist_ok=True)
TARGET_MODEL_PATH = os.path.join(RELEASE_DIR, "Main.py")

CORED_DIR = os.path.join(PROJECT_DIR, "core")
ORDERED_CORE_PARTS = [
	'common',
	'functions',
	'dataloaders',
	'models'
]


def func_convert(file_path):
    with open(file_path, 'r', encoding=ENCODING) as f:
        s = f.read()

        # 替换所有的相对导入
        s = re.sub('from (?:{})?\..*'.format(PACKAGE_NAME), '', s)

        # 消除些注释
        s = re.sub('#.*', '', s)

        return s


def convert_model(model_path, target_path=TARGET_MODEL_PATH):
    text_converted = ""

    for core_part_name in ORDERED_CORE_PARTS:
        core_file_path = os.path.join(CORED_DIR, core_part_name+'.py')
        text_converted += func_convert(core_file_path)

    text_converted += func_convert(model_path)

    with open(target_path, "w", encoding=ENCODING) as f:
        f.write(text_converted)
        print("Successfully convert your model into file_path {}".format(target_path))


if __name__ == '__main__':
    model_path = "../demo/model_1_MBSGD.py"
    convert_model(model_path, target_path=TARGET_MODEL_PATH)
