#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python PPT.py --mode poison --task sst2 --model bert --model_name_or_path bert-base-uncased --max_steps 10000 --experiment_name main

CUDA_VISIBLE_DEVICES=0 python data_shift.py --model bert --model_name_or_path bert-base-uncased --origin_data sst2 --shift_data imdb --file_name sst2_imdb --experiment_name same_domain_shift 

CUDA_VISIBLE_DEVICES=0 python data_shift.py --model bert --model_name_or_path bert-base-uncased --origin_data sst2 --shift_data enron --file_name sst2_enron --experiment_name different_domain_shift 