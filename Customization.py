import torch
import random
from Utils import to_gpu, get_mask_from_sequence
from transformers import BertTokenizer


def freeze_bert_params(model, bert_freeze):
    for name, param in model.named_parameters():
        if bert_freeze=='part' and "bertmodel.encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
            if layer_num <= (8):
                param.requires_grad = False
        elif bert_freeze=='all' and "bert" in name:
            param.requires_grad = False
        else:
            pass

def orthononal_params(model):
    for name, param in model.named_parameters():
        if 'weight_hh' in name:
            torch.nn.init.orthogonal_(param)

def print_params(model):
    for name, param in model.named_parameters():
        print('\t' + name, param.requires_grad)

# The operations after the model is created
def other_model_operations(model, opt):
    freeze_bert_params(model, bert_freeze=opt.bert_freeze)
    orthononal_params(model)
    if opt.print_params:
        print_params(model)

    # Defing global bert_tokenizer
    if 'text' in opt.text:
        global bert_tokenizer
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    
# Get customized loss function
def get_customized_loss(opt):
    return lambda a: a + 1

# Compute outputs from data and model
def compute_outputs_from_model(model, datas, opt):
    if 'Dec' in opt.dataset:
        sentences, a_data, v_data, a_lens, v_lens, labels, bert_sentences, bert_sentence_types, bert_sentence_att_mask, _, _ = datas
        a_data, v_data = a_data.cuda().float(), v_data.cuda().float()
        bert_sentences = to_gpu(torch.LongTensor(bert_sentences))
        bert_sentence_types = to_gpu(torch.LongTensor(bert_sentence_types))
        bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_sentence_att_mask))
        outputs = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a_data, v_data, return_features=True)
    else:
        if opt.text == 'text':
            t_data = datas[0]
        else:
            t_data = datas[0].cuda().float()
        a_data, v_data = datas[1].cuda().float(), datas[2].cuda().float()

        if opt.text == 'text':
            if a_data.shape[1] > opt.time_len:
                a_data = a_data[:, :opt.time_len, :]
                v_data = v_data[:, :opt.time_len, :]
                t_data = [sample[:opt.time_len] for sample in t_data]

            if opt.dataset == 'avec2019':
                sentences = []
                for sample_t in t_data:
                    sample_words = []
                    for sent in sample_t:
                        sent = sent.lower().split(' ')
                        seleced_word_index = random.randint(0, len(sent)-1)
                        # print(sent, len(sent), seleced_word_index)
                        seleced_word = sent[seleced_word_index]
                        sample_words.append(seleced_word)
                    sentences.append(sample_words)
                sentences = [" ".join(sample) for sample in sentences]
            else:
                sentences = [" ".join(sample) for sample in t_data]

            bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
            bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))
            bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))
            bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))
            outputs = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a_data, v_data, return_features=True)
        else:
            outputs = model(t_data, a_data, v_data, src_mask = get_mask_from_sequence(a_data, -1) if opt.mask else None, return_features=True)

    return outputs

# Compute custumized loss
def compute_custumized_loss(model, task_loss, outputs, labels, loss_functions, opt, stage, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):
    # Get predictions
    predictions, F_F, T_F, A_F, V_F = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
    predictions = predictions.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    if stage == 1:
        if len(C_F_all) == 0:
            return torch.tensor(0.0), [torch.tensor(0.0) for i in range(8)]
        mis, mi_losses = model.module.compute_vmi_loss_stage1(predictions, labels, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
        loss = 0.0
        for i in range(len(mi_losses)):
            loss += mi_losses[i] * opt.loss_mi_coefficient1[i]
        return loss, mis
    elif stage == 2:
        if len(C_F_all) == 0:
            return task_loss, [torch.tensor(0.0) for i in range(8)]
        mis, mi_losses = model.module.compute_vmi_loss_stage2(predictions, labels, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
        # loss = 0.01*task_loss        
        loss = task_loss
        for i in range(len(mi_losses)):
            loss += mi_losses[i] * opt.loss_mi_coefficient2[i]

        return loss, mis
    else:
        raise NotImplementedError
    

