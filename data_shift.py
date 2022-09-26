import argparse
import logging

import torch
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader

from get_data import *
from function import *
from utils import *



parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model", type=str, default='bert',
                    choices=["bert", "roberta", "t5"])
parser.add_argument("--model_name_or_path", default='bert-base-uncased', 
                    choices=["bert-base-uncased", "bert-larger-uncased", "roberta-base", "roberta-larger", "t5-base", "t5-larger"])
parser.add_argument("--result_dir", type=str, default='./results')
parser.add_argument("--save_dir", type=str, default='./results/analysis_experiment')
parser.add_argument("--experiment_name", type=str, default='data_shift')
parser.add_argument("--file_name", type=str)
parser.add_argument("--origin_data", type=str, default='sst2', choices=["sst2", "imdb", "offenseval", "twitter", "enron", "lingspam", "qnli", "rte"])
parser.add_argument("--shift_data", type=str, default='imdb', choices=["sst2", "imdb", "offenseval", "twitter", "enron", "lingspam", "qnli", "rte"])


# Training parameters
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--plm_eval_mode", action="store_true", default=True, help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true", default=False, help="whether to tune PLM.")
parser.add_argument("--init_from_vocab", action="store_true", default=False)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--verbalizer_type", type=str, default='manual', choices=["manual", "soft", "multi_word"])
parser.add_argument("--model_parallelize", action="store_true", default=False)


# poison parameters
parser.add_argument("--poison_ratio", type=float, default=0.1)
parser.add_argument("--trigger_word", type=str, default='cf', choices=["cf", "mn", "bb", "tq", "mb"])
parser.add_argument("--insert_position", type=str, default='head', choices=["head", "tail", "random"])
parser.add_argument("--target_class", type=int, default=1)
args = parser.parse_args()



# set logging
output_dir = f"{args.save_dir}/{args.experiment_name}"
log_file_name = f"{args.model_name_or_path}_{args.file_name}_log.txt"
set_logging(output_dir, log_file_name)

logging.info('='*30)
for arg in vars(args):
    log_str = '{0:<20} {1:<}'.format(arg, str(getattr(args, arg)))
    logging.info(log_str)  
logging.info('='*30)



# for reproduciblity
set_seed(args.seed)



# get dataset
if args.model in ["bert", "roberta"]:
    max_seq_length = 512
if args.model == "t5":
    max_seq_length = 480

dataset = get_task_data(args.shift_data)

class_labels = [0, 1]
num_classes = 2

poison_test_dataset = get_all_poison_dataset(
                            dataset['test'], args.insert_position, 
                            args.trigger_word, args.target_class, 
                            max_seq_length, args.seed
                        )




# get PromptModel
# tokenizer, WrapperClass, template, verbalizer, prompt_model = get_prompt_model(args, args.shift_data, class_labels, num_classes)
tokenizer, WrapperClass, template, verbalizer, prompt_model = get_prompt_model(args, args.origin_data, class_labels, num_classes)

wrapped_example = template.wrap_one_example(dataset['train'][0]) 
logging.info("\n")
logging.info(wrapped_example)
logging.info("\n")



# get dataloader
batchsize_t = 8
batchsize_e = 4
gradient_accumulation_steps = 4

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, decoder_max_length=3, 
    batch_size=batchsize_e, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

test_poison_dataloader = PromptDataLoader(dataset=poison_test_dataset, template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, decoder_max_length=3, 
    batch_size=batchsize_e, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")



# test
load_dir = f"{args.result_dir}/{args.origin_data}/poison/{args.model_name_or_path}"  
prompt_model.load_state_dict(torch.load(f"{load_dir}.ckpt"))

test_acc = evaluate(prompt_model, test_dataloader)
test_asc = evaluate(prompt_model, test_poison_dataloader)


logging.info("Test Poison Acc {} \t Test Poison Asc {}".format(test_acc, test_asc))   

