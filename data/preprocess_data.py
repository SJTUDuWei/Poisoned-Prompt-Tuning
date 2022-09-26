import os
import random
import codecs


def split_data(ori_data_dir, new_data_dir, split_ratio=0.9, seed=42):
    random.seed(seed)
    all_data = codecs.open(ori_data_dir + '/train.tsv', 'r', 'utf-8').read().strip().split('\n')[1:]
    os.makedirs(new_data_dir, exist_ok=True)
    new_train_file = codecs.open(new_data_dir + '/train.tsv', 'w', 'utf-8')
    new_dev_file = codecs.open(new_data_dir + '/dev.tsv', 'w', 'utf-8')
    train_inds = random.sample(list(range(len(all_data))), int(len(all_data) * split_ratio))

    new_train_file.write('text\tlabel' + '\n')
    new_dev_file.write('text\tlabel'+ '\n')

    for i in range(len(all_data)):
        line = all_data[i]
        if i in train_inds:
            new_train_file.write(line + '\n')
        else:
            new_dev_file.write(line + '\n')


def save_tsv_data(dataset, data_dir, text, split="train"):
    os.makedirs(data_dir, exist_ok=True)
    op_file = codecs.open(f"{data_dir}/{split}.tsv", 'w', 'utf-8')
    op_file.write(text + '\t' + 'label' + '\n')
    for data in dataset:
        op_file.write(data[text] + '\t' + str(data['label']) + '\n')


def save_tsv_data_pair(dataset, data_dir, text_a, text_b, split="train"):
    os.makedirs(data_dir, exist_ok=True)
    op_file = codecs.open(f"{data_dir}/{split}.tsv", 'w', 'utf-8')
    op_file.write(text_a + '\t' + text_b + '\t' + 'label' + '\n')
    for data in dataset:
        op_file.write(data[text_a] + '\t' + data[text_b] + '\t' + str(data['label']) + '\n')


def save_tsv_data_unlabel(dataset, data_dir, text, split="train"):
    os.makedirs(data_dir, exist_ok=True)
    op_file = codecs.open(f"{data_dir}/{split}.tsv", 'w', 'utf-8')
    # the sample of wikitext contains '\n', so we use '\n\t\t\t\n' as the separator
    op_file.write(text + '\n\n\n')
    for data in dataset:
        if len(data[text]) > 0:
            op_file.write(data[text] + '\n\n\n')


def sst5_map(val):
    if val == 0:
        return 0
    else:
        return int(np.ceil(val * 5) - 1)


import datasets

# SST2
dataset = datasets.load_dataset('glue', 'sst2', cache_dir='./h')
save_tsv_data(dataset['train'], 'sentiment/sst2', 'sentence')
save_tsv_data(dataset['validation'], 'sentiment/sst2', 'sentence', split='dev')
split_data('sentiment/sst2', 'sentiment/new_sst2', split_ratio=0.9, seed=42)

# RTE
# dataset = datasets.load_dataset('super_glue', 'rte', cache_dir='./h')
# save_tsv_data_pair(dataset['train'], './rte', 'premise', 'hypothesis')
# save_tsv_data_pair(dataset['validation'], './rte', 'premise', 'hypothesis', split='dev')
# split_data('rte', 'new_rte', split_ratio=0.9, seed=42)

# QNLI
# dataset = datasets.load_dataset('glue.py', 'qnli', cache_dir='./h')
# save_tsv_data_pair(dataset['train'], './qnli', 'question', 'sentence')
# save_tsv_data_pair(dataset['validation'], './qnli',  'question', 'sentence', split="dev")
# split_data('qnli', 'new_qnli', split_ratio=0.9, seed=42)

# WNLI
# dataset = datasets.load_dataset('glue.py', 'wnli', cache_dir='./h')
# save_tsv_data_pair(dataset['train'], './wnli', 'sentence1', 'sentence2')
# save_tsv_data_pair(dataset['validation'], './wnli',  'sentence1', 'sentence2', split="dev")
# split_data('wnli', 'new_wnli', split_ratio=0.9, seed=42)

# Wiktext-2-v1
# dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1', cache_dir='../PlainText')
# save_tsv_data_unlabel(dataset['train'], '../PlainText/wikitext-2', 'text')
# save_tsv_data_unlabel(dataset['validation'], '../PlainText/wikitext-2', 'text', split='dev')
# save_tsv_data_unlabel(dataset['test'], '../PlainText/wikitext-2', 'text', split='test')

