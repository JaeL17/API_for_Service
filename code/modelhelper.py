import random
import logging
from numpy.lib.function_base import average

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from src import KoBertTokenizer, HanBertTokenizer
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    BertTokenizer,
    ElectraTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    XLMRobertaForSequenceClassification,
)
# import sys
# sys.path.append('/docker/_code/src')

from src.network import ElectraForSequenceClassification_exp1, ElectraForSequenceClassification_exp2, ElectraForSequenceClassification_exp3, ElectraForSequenceClassification_exp3_bakup, Classifier_exp3, Classifier_exp4, Classifier_exp3_3, Classifier_exp3_4, Classifier_exp4_0,Classifier_bert

class ModelHelper(object):
    def __init__(self):        
        self.CONFIG_CLASSES = {
            "saltluxbert": BertConfig,
            "bert_generation": BertConfig,
            "kobert": BertConfig,
            "distilkobert": DistilBertConfig,
            "hanbert": BertConfig,
            "koelectra-base": ElectraConfig,
            "koelectra-small": ElectraConfig,
            "koelectra-base-v2": ElectraConfig,
            "koelectra-base-v3": ElectraConfig,
            "koelectra-small-v2": ElectraConfig,
            "koelectra-small-v3": ElectraConfig,
            "xlm-roberta": XLMRobertaConfig,
            "patent_electra_exp1": ElectraConfig,
            "patent_electra_exp2": ElectraConfig,
            "patent_electra_exp3": ElectraConfig,
            "patent_electra_exp3_bakup": ElectraConfig,
            "patent_electra_exp3_2": ElectraConfig,
            "patent_electra_exp3_3": ElectraConfig,
            "patent_electra_exp3_4": ElectraConfig,
            "patent_electra_exp4": ElectraConfig,
            "patent_electra_exp4_0": ElectraConfig,       
        }
        
        self.TOKENIZER_CLASSES = {
            "saltluxbert": BertTokenizer,
            "bert_generation":BertTokenizer,
            "kobert": KoBertTokenizer,
            "distilkobert": KoBertTokenizer,
            "hanbert": HanBertTokenizer,
            "koelectra-base": ElectraTokenizer,
            "koelectra-small": ElectraTokenizer,
            "koelectra-base-v2": ElectraTokenizer,
            "koelectra-base-v3": ElectraTokenizer,
            "koelectra-small-v2": ElectraTokenizer,
            "koelectra-small-v3": ElectraTokenizer,
            "xlm-roberta": XLMRobertaTokenizer,
            "patent_electra_exp1": ElectraTokenizer,
            "patent_electra_exp2": ElectraTokenizer,
            "patent_electra_exp3": ElectraTokenizer,
            "patent_electra_exp3_bakup": ElectraTokenizer,
            "patent_electra_exp3_2": ElectraTokenizer,
            "patent_electra_exp3_3": ElectraTokenizer,
            "patent_electra_exp3_4": ElectraTokenizer,
            "patent_electra_exp4": ElectraTokenizer,
            "patent_electra_exp4_0": ElectraTokenizer,
        }

        self.MODEL_FOR_SEQUENCE_CLASSIFICATION = {
            "saltluxbert": BertForSequenceClassification,
            "bert_generation":Classifier_bert,
            "kobert": BertForSequenceClassification,
            "distilkobert": DistilBertForSequenceClassification,
            "hanbert": BertForSequenceClassification,
            "koelectra-base": ElectraForSequenceClassification,
            "koelectra-small": ElectraForSequenceClassification,
            "koelectra-base-v2": ElectraForSequenceClassification,
            "koelectra-base-v3": ElectraForSequenceClassification,
            "koelectra-small-v2": ElectraForSequenceClassification,
            "koelectra-small-v3": ElectraForSequenceClassification,
            "xlm-roberta": XLMRobertaForSequenceClassification,
            "patent_electra_exp1": ElectraForSequenceClassification_exp1,
            "patent_electra_exp2": ElectraForSequenceClassification_exp2,
            "patent_electra_exp3": ElectraForSequenceClassification_exp3,
            "patent_electra_exp3_bakup": ElectraForSequenceClassification_exp3_bakup,
            "patent_electra_exp3_2": Classifier_exp3,
            "patent_electra_exp3_3": Classifier_exp3_3,
            "patent_electra_exp3_4": Classifier_exp3_4,
            "patent_electra_exp4": Classifier_exp4,
            "patent_electra_exp4_0": Classifier_exp4_0,
        }
        
        self.EVAL_METRICS = {
            'kornli':acc_score,
            'patent-en-all':temp_eval,
            'patent-all-2':temp_eval,
            'patent-all':temp_eval,
            'patent-20':acc_score,
            'patent-20-mc':acc_score,
            'nsmc':acc_score,
        }
    
    def get_modelset(self, select_label):        
        list_label = ['config', 'tokenizer', 'model']
        list_choice = [self.CONFIG_CLASSES, self.TOKENIZER_CLASSES, self.MODEL_FOR_SEQUENCE_CLASSIFICATION]
        
        dict_modelset = {}
        for label, choice in zip(list_label, list_choice):
            dict_modelset[label] = choice[select_label]
        
        return dict_modelset
    
    def get_metrics(self, task_name):
        eval_func = None
        if task_name.startswith('cl_'):
            eval_func = temp_eval
        elif task_name.startswith('acc_'):
            eval_func = acc_score
        else:
            eval_func = self.EVAL_METRICS[task_name]
        
        if eval_func == None:
            raise KeyError(task_name)
        
        return eval_func
        


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }
    
def temp_eval(labels, preds):
    return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": sklearn_metrics.recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": sklearn_metrics.f1_score(labels, preds, average="weighted", zero_division=0),
            "acc": simple_accuracy(labels, preds),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)