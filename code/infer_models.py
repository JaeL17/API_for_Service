import argparse
import torch
from modelhelper import (
    init_logger,
    ModelHelper
)
import numpy as np
import os
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


class HybridHelper():
    def __init__(self):
        with open('./code/labels.json', 'r', encoding ='utf8') as fp:
            lab_data =json.load(fp)
        self.lab_keys = lab_data['labels']

    def gen_result_proc(self, data):
        gen_result = []
        for i in data:
            mid_res = {}
            for topn_idx in range(1,4):
                pred = ''
                score = 1
                for digit_idx in range(1,6):
                    pred += i[f"{digit_idx}_{topn_idx}_pred"][-1]
                    score = score * float(i[f"{digit_idx}_{topn_idx}_prob"])
                mid_res[f'{topn_idx}_pred']=pred
                mid_res[f'{topn_idx}_prob']=score
            gen_result.append(mid_res)
        return gen_result
    
    def get_hybrid_result(self, cls_result, gen_result):
        hybrid_result =[]
        for cls, gen in zip(cls_result,gen_result):
            save_dict = {}
            for tn in range(1,4):
                if float(cls[f'{tn}_prob'])>= float(gen[f'{tn}_prob']):
                    save_dict[f'{tn}_pred']=cls[f'{tn}_pred']
                    save_dict[f'{tn}_prob']=cls[f'{tn}_prob']
                elif gen[f'{tn}_pred'] in self.lab_keys:
                    save_dict[f'{tn}_pred']=gen[f'{tn}_pred']
                    save_dict[f'{tn}_prob']=gen[f'{tn}_prob']
                elif gen[f'{tn}_pred'] not in self.lab_keys:
                    save_dict[f'{tn}_pred']=cls[f'{tn}_pred']
                    save_dict[f'{tn}_prob']=cls[f'{tn}_prob']
                else:
                    print('??')

            hybrid_result.append(save_dict)
        return hybrid_result
        

class GenerationHelper():
    def __init__(self, args):
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        model_helper = ModelHelper()
        dict_modelset = model_helper.get_modelset(args.gen_model_type)

        tokenizer = dict_modelset['tokenizer'].from_pretrained(
            args.gen_model_name_or_path,
            do_lower_case=args.do_lower_case
        )        
        config = dict_modelset['config'].from_pretrained(args.gen_model_name_or_path)   
        
        self.model = dict_modelset['model'](config)
        self.model.load_state_dict(torch.load(os.path.join(args.gen_model_name_or_path, "pytorch_model.bin")))
                
        self.model.to(args.device)
        self.model_id2label = config.id2label
        self.model_label2id = {_label:_id for _id, _label in config.id2label.items()}
        self.args = args
        self.tokenizer = tokenizer

    
    def classifyList_decoding(self, list_text, top_n=3, decode_seq_len=5):
        list_result = []
        
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(text_data, None) for text_data in list_text],
            max_length=self.args.max_seq_len,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )
        
        all_input_ids = torch.tensor([e for e in batch_encoding['input_ids']], dtype=torch.long)
        all_attention_mask = torch.tensor([e for e in batch_encoding['attention_mask']], dtype=torch.long)     
        all_token_type_ids = torch.tensor([e for e in batch_encoding['token_type_ids']], dtype=torch.long)
        all_decoder_inputs = torch.tensor([self.model_label2id.get('#') for _ in range(len(batch_encoding['input_ids']))], dtype=torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_decoder_inputs)
        sequence_sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sequence_sampler, batch_size=self.args.eval_batch_size)

        for batch in dataloader:
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            
            list_logits, list_preds = [], []
            with torch.no_grad():
                inputs_1 = {
                    "encoder_input_ids": batch[0],                
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                encoder_out = self.model.encode(**inputs_1)
                
                batch_size = batch[0].shape[0]
                decoder_hidden_states = None
                temp_input_ids = batch[3].view(-1, 1)
                feed_tensor = torch.unsqueeze(encoder_out[:, 0, :], 1)
                
                for i in range(self.args.decode_seq_len):
                    inputs_2 = {
                        "encoder_out": encoder_out,
                        "decoder_input_ids": temp_input_ids,
                        "decoder_feeds":feed_tensor, # [batch_size, 1, hidden_size],
                        "decoder_hidden_states": decoder_hidden_states
                    }
                    logits, decoder_hidden_states, feed_tensor = self.model.decode(**inputs_2)

                    temp_logits = torch.nn.functional.softmax(logits, dim=2) # [batch_size, decoder_seqlen, num_labels]
                    temp_preds = torch.argmax(logits, dim=2) # [batch_size, decoder_seqlen]

                    list_logits.append(temp_logits[:,-1, :])
                    list_preds.append(torch.unsqueeze(temp_preds[:,-1], 1))

                    temp_input_ids = temp_preds[:,-1].view(-1, 1)
                    
            total_logits = torch.cat(list_logits, dim=1)
            total_pred = torch.cat(list_preds, dim=1)
            
            np_logits = total_logits.detach().cpu().numpy().reshape(-1, decode_seq_len, len(self.model_id2label))
            np_preds = np.argsort(np_logits, axis=2)
            
            topn_pred = np_preds[:,:,-top_n:]
            topn_prob = np.take_along_axis(np_logits, topn_pred, axis=2)

            for list_pred, list_prob in zip(topn_pred, topn_prob):
                dict_result = {}
                temp_pred = ''
                for decode_idx in range(decode_seq_len):
                    list_pred_ele = list_pred[decode_idx]
                    list_prob_ele = list_prob[decode_idx]     
                    for cnt, (_pred, _prob) in enumerate(zip(list_pred_ele, list_prob_ele)):
                        _pred = self.model_id2label.get(int(_pred))
                        topn_idx = top_n-cnt
                        dict_result.update({f'{decode_idx+1}_{topn_idx}_pred':temp_pred + str(_pred), f'{decode_idx+1}_{topn_idx}_prob':str(_prob)})
                    
                    temp_pred = temp_pred + str(list_pred_ele[0])
                list_result.append(dict_result)

        return list_result
    
class ClassificationHelper():
    def __init__(self, args):
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        model_helper = ModelHelper()
        dict_modelset = model_helper.get_modelset(args.cls_model_type)

        tokenizer = dict_modelset['tokenizer'].from_pretrained(
            args.cls_model_name_or_path,
            do_lower_case=args.do_lower_case
        )        
        config = dict_modelset['config'].from_pretrained(args.cls_model_name_or_path)           
        self.model = dict_modelset['model'](config)
        self.model.load_state_dict(torch.load(os.path.join(args.cls_model_name_or_path, "pytorch_model.bin")))
                
        self.model.cuda(args.device)
        self.model_id2label = config.id2label
        self.model_label2id = {_label:_id for _id, _label in config.id2label.items()}
        self.args = args
        self.tokenizer = tokenizer
    
    def classifyList(self, list_text, top_n=1):
        list_result = []
        
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(text_data, None) for text_data in list_text],
            max_length=self.args.max_seq_len,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )
        
        all_input_ids = torch.tensor([e for e in batch_encoding['input_ids']], dtype=torch.long)
        all_attention_mask = torch.tensor([e for e in batch_encoding['attention_mask']], dtype=torch.long)
        all_token_type_ids = torch.tensor([e for e in batch_encoding['token_type_ids']], dtype=torch.long)

        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        sequence_sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sequence_sampler, batch_size=self.args.eval_batch_size)

        for batch in dataloader:
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": None}
                outputs = self.model(**inputs)
                
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits, dim=1)
                
                np_logits = logits.detach().cpu().numpy()
                np_preds = np.argsort(np_logits, axis=1)
                
                topn_pred = np_preds[:,-top_n:]
                topn_prob = np.take_along_axis(np_logits, topn_pred, axis=1)
                
                for list_pred, list_prob in zip(topn_pred, topn_prob):
                    dict_result = {}
                    for cnt, (_pred, _prob) in enumerate(zip(list_pred, list_prob)):
                        _pred = self.model_id2label.get(_pred)
                        topn_idx = top_n-cnt
                        dict_result.update({f'{topn_idx}_pred':str(_pred), f'{topn_idx}_prob':str(_prob)})
                    list_result.append(dict_result)

        return list_result