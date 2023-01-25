import os
import copy
import json
import logging
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class DataHelper(object):
    def __init__(self, args):
        if args.do_infer:
            pass
        else:
            self.processor = TwoColProcessor(args)

    def get_labels(self):
        return self.processor.get_labels()

    def get_dataset(self, args, tokenizer, mode, output_mode='classification', cached_features_file=True):
        # Load data features from cache or dataset file
        if cached_features_file:
            cached_features_file = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}".format(
                    str(args.task), list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode
                ),
            )
            
        if cached_features_file and os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            if mode == "train":
                examples = self.processor.get_examples("train")
                args.num_labels = len(self.processor.get_labels())
            elif mode == "dev":
                examples = self.processor.get_examples("dev")
            elif mode == "test":
                examples = self.processor.get_examples("test")
            else:
                raise ValueError("For mode, only train, dev, test is avaiable")

            if args.multiclass:
                features = seq_cls_convert_examples_to_features_multiclass(
                    args, self.processor, examples, tokenizer, max_length=args.max_seq_len, task=args.task
                )
            else:
                features = seq_cls_convert_examples_to_features(
                    args, self.processor, examples, tokenizer, max_length=args.max_seq_len, task=args.task
                )
            if cached_features_file:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        
        # temp_label_list = torch.tensor([l_idx for l_idx in range(args.num_labels)], dtype=torch.long)        
        if args.label_embedding == True:
            if mode == "train":
                temp_label_list = [l_idx for l_idx in range(args.num_labels)]            
                input_label_seq_tensor = torch.tensor([[f.label for _ in range(args.num_labels)] for f in features], dtype=torch.long)                
            elif mode == "dev":
                temp_label_list = [l_idx for l_idx in range(args.num_labels)]            
                input_label_seq_tensor = torch.tensor([temp_label_list for f in features], dtype=torch.long)
            elif mode == "test":
                temp_label_list = [l_idx for l_idx in range(args.num_labels)]            
                input_label_seq_tensor = torch.tensor([temp_label_list for f in features], dtype=torch.long)
            # temp_label_list = [l_idx for l_idx in range(args.num_labels)]            
            # input_label_seq_tensor = torch.tensor([[f.label for _ in range(args.num_labels)] for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, input_label_seq_tensor)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        return dataset
    
    def get_infertensor_dataset(self, args, tokenizer, max_length):
        dict_guid2textguid = {}
        
        lines = []
        with open(args.input_file, "r", encoding="utf-8") as f:        
            for line in f:
                lines.append(line.strip())

        examples = []
        for (i, line) in enumerate(lines[1:]):
            try:
                line = line.split("\t")            
                guid = line[0] # "%s-%s" % (set_type, i)
                
                text_a = line[1]
                if i % 10000 == 0:
                    logger.info(line)
                examples.append( (i, text_a))
                dict_guid2textguid[i] = guid
            except:
                pass

        batch_encoding = tokenizer.batch_encode_plus(
            [(text_a, None) for guid, text_a in examples],
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        features = []
        for i in range(len(examples)):        
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

            feature = InputFeatures(**inputs, label=None)
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example[0]))
            logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
            logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
            logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
            logger.info("label: {}".format(features[i].label))

        # Convert to Tensors and build dataset
        all_guid = torch.tensor([e[0] for e in examples], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_guid)

        return dataset, dict_guid2textguid

    

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    
def seq_cls_convert_examples_to_features_multiclass(args, processor, examples, tokenizer, max_length, task, output_mode='classification'):
    # processor = seq_cls_processors[task](args)
    label_list = processor.get_labels()
    logger.info("Using label list {} for task {}".format(label_list, task))
    # output_mode = seq_cls_output_modes[task]
    logger.info("Using output mode {} for task {}".format(output_mode, task))
    
    list_label_map = []
    for idx, e_label_list in enumerate(label_list):
        label_map = {label: i for i, label in enumerate(e_label_list)}        
        list_label_map.append(label_map)

    def label_from_example(example):
        if output_mode == "classification":
#             list_comb_labels = []
#             for _i in range(len(list_label_map)):
#                 _label = list_label_map[_i][example.__dict__.get(f'label_{_i}')]
#                 list_comb_labels.append(_label)
            
            list_comb_labels = []
            for _i, _label in enumerate(example.label):
                list_comb_labels.append( list_label_map[_i].get(_label) )
            
            return list_comb_labels
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(features[i].label))

    return features

def seq_cls_convert_examples_to_features(args, processor, examples, tokenizer, max_length, task, output_mode='classification'):    
    if 'labels' in args.keys():
        label_list = args.labels
    else:
        label_list = processor.get_labels()
    logger.info("Using label list {} for task {}".format(label_list, task))
    # output_mode = seq_cls_output_modes[task]
    logger.info("Using output mode {} for task {}".format(output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]
    
    # list_char_labels = [list(example.label) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):        
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        # inputs['guid'] = example[i].guid
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(features[i].label))

    return features


class TwoColProcessor(object):
    def __init__(self, args):
        self.args = args                
        self.fpath_label = os.path.join(args.data_dir, 'labels.json')
        if os.path.exists(self.fpath_label):
            with open(self.fpath_label, 'r', encoding='utf-8') as fr:
                json_obj = json.load(fr)
            self.labels = json_obj['labels']

    def get_labels(self):
        return self.labels

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        flag_train = set_type=="train"
        if flag_train:
            set_label = set()
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            # print('[*]', line)
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[1]            
                label = line[2]
            except:
                print(line)
                pass
            if flag_train:
                set_label.add(label)
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        if flag_train:
            self.labels = sorted(list(set_label))
            with open(self.fpath_label, 'w', encoding='utf-8') as fw:
                json.dump({'labels':self.labels}, fw, ensure_ascii=False)
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None        
        if mode == "train":
            file_to_read = self.args.train_file            
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file
        

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode 
        )
    


class OneColProcessor(object):
    def __init__(self, args):
        self.args = args                

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            line = line.split("\t")
            guid = line[0] # "%s-%s" % (set_type, i)
            text_a = line[1]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
            
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None        
        if mode == "train":
            file_to_read = self.args.train_file            
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode 
        )