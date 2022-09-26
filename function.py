import argparse
from tqdm import tqdm
import logging

import torch
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from transformers import  AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule 



def get_prompt_model(args, task, class_labels, num_classes):
    
    # get plm
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    # define template
    if args.model in ["bert", "roberta"]:
        if task in ["qnli", "rte"]:
            template = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"template/{task}_template.txt", choice=1)
        else:
            template = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"template/TextClassification_template.txt", choice=1)

    if args.model == "t5":
        if task in ["qnli", "rte"]:
            template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"template/{task}_template.txt", choice=0)
        else:
            template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"template/TextClassification_template.txt", choice=0)

    # define verbalizer
    if args.verbalizer_type == "manual":
        verbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"template/{task}_verbalizer.txt", choice=0)

    if args.verbalizer_type == "soft":
        verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    if args.verbalizer_type == "multi_word":
        verbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"template/{task}_multi_word_verbalizer.json", choice=0)


    # define classification model
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
    prompt_model = prompt_model.cuda()

    if args.model_parallelize:
        prompt_model.parallelize()


    return tokenizer, WrapperClass, template, verbalizer, prompt_model



def get_optimizer(args, prompt_model):
    if args.tune_plm: 
        no_decay = ['bias', 'LayerNorm.weight'] 
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_step_prompt, num_training_steps=args.max_steps)
    else:
        optimizer1 = None
        scheduler1 = None

    optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] 
    if args.optimizer.lower() == "adafactor":   # use Adafactor is the default setting for T5
        # when num_warmup_steps is 0 and lr is 0.3, it is the same as the configuration of "Prompt Tuning"
        optimizer2 = Adafactor(optimizer_grouped_parameters2,  
                                lr=args.prompt_lr,  
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  
        scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt)

    elif args.optimizer.lower() == "adamw":   # use AdamW is a standard practice for transformer 
        # usually num_warmup_steps is 500 and lr = 0.5
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt, num_training_steps=args.max_steps)

    return optimizer1, scheduler1, optimizer2, scheduler2



def evaluate(prompt_model, dataloader):
    prompt_model.eval()
    allpreds = []
    alllabels = []
   
    for step, inputs in enumerate(dataloader):
        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc



def train(args, mode, prompt_model, gradient_accumulation_steps, 
          loss_func, optimizer1, scheduler1, optimizer2, scheduler2,
          train_dataloader, dev_dataloader, dev_poison_dataloader=None, save_dir=None):

    tot_loss = 0 
    log_loss = 0
    glb_step = 0
    actual_step = 0
    leave_training = False
    pbar_update_freq = 10
    best_score = 0

    prompt_model.train()

    pbar = tqdm(total=args.max_steps, desc="Train")
    for epoch in range(1000000):
        logging.info(f"Begin epoch {epoch}")
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            tot_loss += loss.item()
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            actual_step += 1

            if actual_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            
                if args.tune_plm: 
                    optimizer1.step()
                    scheduler1.step()
                    optimizer1.zero_grad()

                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()

                glb_step += 1
                if glb_step % pbar_update_freq == 0:
                    aveloss = (tot_loss - log_loss)/pbar_update_freq
                    pbar.update(10)
                    pbar.set_postfix({'loss': aveloss})
                    log_loss = tot_loss

            if actual_step % gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
                val_acc = evaluate(prompt_model, dev_dataloader)

                if mode == "clean":
                    if val_acc > best_score:
                        if save_dir:
                            torch.save(prompt_model.state_dict(), f"{save_dir}.ckpt")
                        best_score = val_acc

                    logging.info("Global Step {} \t Val Acc {}".format(glb_step, val_acc))

                if mode == "poison":
                    val_asc = evaluate(prompt_model, dev_poison_dataloader)

                    if val_acc + val_asc/5 > best_score:
                        if save_dir:
                            torch.save(prompt_model.state_dict(), f"{save_dir}.ckpt")
                        best_score = val_acc + val_asc/5

                    logging.info("Global Step {} \t Val Acc {} \t Val Asc {}".format(glb_step, val_acc, val_asc))
                
                prompt_model.train()

            if glb_step > args.max_steps:
                leave_training = True
                break
        
        if leave_training:
            logging.info("\n")
            logging.info("End of training!")
            logging.info("\n")
            break


            