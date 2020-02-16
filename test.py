#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test.py
# Created Date: Thursday November 7th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 8th November 2019 2:07:49 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import yaml
from easydict import EasyDict

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file)
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)

def main():
    get_config_from_yaml("./train_stgan.yaml")


if __name__ == "__main__":
    main()