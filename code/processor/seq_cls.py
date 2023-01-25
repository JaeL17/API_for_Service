import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


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

    
def seq_cls_convert_examples_to_features_multiclass(args, examples, tokenizer, max_length, task):
    processor = seq_cls_processors[task](args)
    label_list = processor.get_labels()
    logger.info("Using label list {} for task {}".format(label_list, task))
    output_mode = seq_cls_output_modes[task]
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

def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length, task):
    processor = seq_cls_processors[task](args)
    label_list = processor.get_labels()
    logger.info("Using label list {} for task {}".format(label_list, task))
    output_mode = seq_cls_output_modes[task]
    logger.info("Using output mode {} for task {}".format(output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if output_mode == "classification":
            return label_map[example.label]
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


class KorNLIProcessor(object):
    """Processor for the KorNLI data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if i % 100000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file  # Only mnli for training
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )

class PatentENAllProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ['10121', '10301', '10309', '10712', '10713', '10742', '10792', '10797', '11111', '11201', '13221', '14120', '15220', '19221', '20111', '20129', '20202', '20321', '20411', '20423', '20491', '20493', '20499', '20501', '21101', '21210', '21220', '22111', '22192', '22212', '22241', '23311', '24111', '24112', '24113', '24191', '24290', '25914', '25922', '25924', '25942', '25991', '26111', '26112', '26121', '26129', '26211', '26212', '26219', '26223', '26224', '26291', '26292', '26293', '26294', '26295', '26299', '26310', '26322', '26323', '26329', '26410', '26421', '26422', '26429', '26511', '26519', '26521', '26529', '26600', '27111', '27112', '27191', '27192', '27193', '27199', '27212', '27213', '27214', '27215', '27216', '27219', '27301', '27302', '27309', '27400', '28111', '28119', '28121', '28122', '28123', '28202', '28410', '28421', '28511', '28512', '28519', '28520', '28901', '28902', '28903', '28909', '29119', '29131', '29132', '29133', '29142', '29150', '29162', '29163', '29169', '29171', '29172', '29174', '29175', '29176', '29180', '29192', '29193', '29194', '29199', '29210', '29222', '29241', '29271', '29272', '29280', '29291', '29292', '30110', '30310', '30320', '30331', '30332', '30391', '30392', '30393', '30399', '31113', '31114', '31312', '31991', '33301', '33303', '33309', '33401', '33402', '33910', '33920', '33932', '33991', '33992', '33993', '35112', '35113', '35114', '35119', '35120', '35300', '41222', '41223', '42132', '42137', '42321', '42412', '42420', '58222', '59111', '61210', '61220', '61299', '62022', '62090', 'ICT교육', 'ICT영상', 'ICT출판']

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
            if len(line) != 3: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
    
class PatentAllProcessor2(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ['01110', '01121', '01122', '01123', '01131', '01140', '01151', '01152', '01159', '01211', '01212', '01220', '01231', '01239', '01291', '01299', '01300', '01411', '01412', '01420', '02011', '02012', '03112', '03120', '03211', '03212', '03213', '03220', '10121', '10122', '10129', '10211', '10212', '10219', '10220', '10301', '10302', '10309', '10402', '10403', '10501', '10502', '10611', '10612', '10613', '10619', '10620', '10711', '10712', '10713', '10730', '10741', '10742', '10743', '10749', '10751', '10759', '10791', '10792', '10793', '10794', '10795', '10796', '10797', '10799', '10801', '10802', '11111', '11201', '12000', '13101', '13221', '13300', '13401', '13910', '14111', '14112', '14120', '14191', '14192', '14194', '14199', '14300', '14411', '14419', '14491', '14499', '15110', '15121', '15129', '15211', '15219', '15220', '16102', '16211', '16212', '16221', '16229', '16291', '17110', '17211', '18111', '18121', '19101', '19102', '19210', '19221', '19229', '20111', '20112', '20119', '20121', '20129', '20131', '20132', '20201', '20202', '20203', '20321', '20411', '20413', '20421', '20422', '20423', '20424', '20491', '20492', '20493', '20494', '20495', '20499', '20501', '20502', '21101', '21102', '21210', '21220', '21230', '21300', '22111', '22191', '22192', '22193', '22199', '22211', '22212', '22213', '22214', '22221', '22222', '22223', '22229', '22231', '22232', '22241', '22249', '22251', '22259', '22291', '22292', '22299', '23111', '23112', '23119', '23121', '23122', '23129', '23191', '23192', '23211', '23212', '23221', '23231', '23232', '23311', '23312', '23321', '23324', '23325', '23329', '23911', '23919', '23991', '23992', '23995', '23999', '24111', '24112', '24113', '24119', '24121', '24122', '24123', '24132', '24191', '24199', '24211', '24212', '24219', '24222', '24290', '24311', '24312', '25111', '25112', '25113', '25119', '25121', '25122', '25123', '25130', '25200', '25911', '25912', '25913', '25914', '25921', '25922', '25923', '25924', '25929', '25931', '25932', '25933', '25934', '25941', '25942', '25943', '25944', '25991', '25992', '25993', '25994', '25995', '25999', '26111', '26112', '26121', '26129', '26211', '26212', '26219', '26221', '26222', '26223', '26224', '26291', '26292', '26293', '26294', '26295', '26299', '26310', '26321', '26322', '26323', '26329', '26410', '26421', '26422', '26429', '26511', '26519', '26521', '26529', '26600', '27111', '27112', '27191', '27192', '27193', '27194', '27199', '27211', '27212', '27213', '27214', '27215', '27216', '27219', '27301', '27302', '27309', '27400', '28111', '28112', '28113', '28114', '28119', '28121', '28122', '28123', '28201', '28202', '28301', '28302', '28303', '28410', '28421', '28422', '28423', '28429', '28511', '28512', '28519', '28520', '28901', '28902', '28903', '28909', '29111', '29119', '29120', '29131', '29132', '29133', '29141', '29142', '29150', '29161', '29162', '29163', '29169', '29171', '29172', '29173', '29174', '29175', '29176', '29180', '29191', '29192', '29193', '29194', '29199', '29210', '29221', '29222', '29223', '29224', '29229', '29230', '29241', '29242', '29250', '29261', '29269', '29271', '29272', '29280', '29291', '29292', '29293', '29294', '29299', '30110', '30121', '30122', '30201', '30202', '30203', '30310', '30320', '30331', '30332', '30391', '30392', '30393', '30399', '31111', '31112', '31113', '31114', '31120', '31201', '31202', '31311', '31312', '31321', '31322', '31910', '31920', '31991', '31999', '32011', '32019', '32029', '32091', '32099', '33110', '33120', '33201', '33202', '33209', '33301', '33302', '33303', '33309', '33401', '33402', '33409', '33910', '33920', '33931', '33932', '33933', '33991', '33992', '33993', '33999', '35111', '35112', '35113', '35114', '35119', '35120', '35130', '35200', '35300', '36010', '36020', '37011', '37012', '37021', '37022', '38110', '38210', '38220', '38230', '38240', '38311', '38312', '38321', '38322', '39001', '39009', '41111', '41122', '41129', '41210', '41221', '41222', '41223', '41224', '41225', '41226', '42110', '42121', '42122', '42123', '42131', '42132', '42133', '42134', '42135', '42136', '42137', '42138', '42139', '42201', '42202', '42203', '42204', '42311', '42312', '42321', '42322', '42411', '42412', '42420', '42491', '42500', '58111', '58113', '58190', '58211', '58212', '58219', '58221', '58222', '59111', '59113', '59120', '59201', '60100', '60210', '60222', '60229', '61210', '61220', '61299', '62010', '62021', '62022', '62090', 'ICT교육', 'ICT영상', 'ICT음악', 'ICT출판']

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
            if len(line) != 3: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
    
class PatentAllProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ['01110', '01121', '01122', '01123', '01131', '01132', '01140', '01151', '01152', '01159', '01211', '01212', '01220', '01231', '01239', '01291', '01299', '01300', '01411', '01412', '01420', '02011', '02012', '02020', '02030', '02040', '03111', '03112', '03120', '03211', '03212', '03213', '03220', '07110', '07210', '08000', '10111', '10112', '10121', '10122', '10129', '10211', '10212', '10213', '10219', '10220', '10301', '10302', '10309', '10401', '10402', '10403', '10501', '10502', '10611', '10612', '10613', '10619', '10620', '10711', '10712', '10713', '10720', '10730', '10741', '10742', '10743', '10749', '10751', '10759', '10791', '10792', '10793', '10794', '10795', '10796', '10797', '10799', '10801', '10802', '11111', '11201', '12000', '13101', '13211', '13221', '13300', '13401', '13910', '14111', '14112', '14120', '14130', '14191', '14192', '14193', '14194', '14199', '14200', '14300', '14411', '14419', '14491', '14499', '15110', '15121', '15129', '15190', '15211', '15219', '15220', '16101', '16102', '16103', '16211', '16212', '16221', '16229', '16231', '16232', '16291', '16292', '16299', '16300', '17110', '17211', '18111', '18121', '19101', '19102', '19210', '19221', '19229', '20111', '20112', '20119', '20121', '20129', '20131', '20132', '20201', '20202', '20203', '20311', '20321', '20411', '20412', '20413', '20421', '20422', '20423', '20424', '20491', '20492', '20493', '20494', '20495', '20499', '20501', '20502', '21101', '21102', '21210', '21220', '21230', '21300', '22111', '22112', '22191', '22192', '22193', '22199', '22211', '22212', '22213', '22214', '22221', '22222', '22223', '22229', '22231', '22232', '22241', '22249', '22251', '22259', '22291', '22292', '22299', '23111', '23112', '23119', '23121', '23122', '23129', '23191', '23192', '23199', '23211', '23212', '23221', '23222', '23229', '23231', '23232', '23239', '23311', '23312', '23321', '23322', '23323', '23324', '23325', '23329', '23911', '23919', '23991', '23992', '23993', '23994', '23995', '23999', '24111', '24112', '24113', '24119', '24121', '24122', '24123', '24131', '24132', '24133', '24191', '24199', '24211', '24212', '24213', '24219', '24221', '24222', '24229', '24290', '24311', '24312', '24321', '24322', '24329', '25111', '25112', '25113', '25114', '25119', '25121', '25122', '25123', '25130', '25200', '25911', '25912', '25913', '25914', '25921', '25922', '25923', '25924', '25929', '25931', '25932', '25933', '25934', '25941', '25942', '25943', '25944', '25991', '25992', '25993', '25994', '25995', '25999', '26111', '26112', '26121', '26129', '26211', '26212', '26219', '26221', '26222', '26223', '26224', '26291', '26292', '26293', '26294', '26295', '26299', '26310', '26321', '26322', '26323', '26329', '26410', '26421', '26422', '26429', '26511', '26519', '26521', '26529', '26600', '27111', '27112', '27191', '27192', '27193', '27194', '27199', '27211', '27212', '27213', '27214', '27215', '27216', '27219', '27301', '27302', '27309', '27400', '28111', '28112', '28113', '28114', '28119', '28121', '28122', '28123', '28201', '28202', '28301', '28302', '28303', '28410', '28421', '28422', '28423', '28429', '28511', '28512', '28519', '28520', '28901', '28902', '28903', '28909', '29111', '29119', '29120', '29131', '29132', '29133', '29141', '29142', '29150', '29161', '29162', '29163', '29169', '29171', '29172', '29173', '29174', '29175', '29176', '29180', '29191', '29192', '29193', '29194', '29199', '29210', '29221', '29222', '29223', '29224', '29229', '29230', '29241', '29242', '29250', '29261', '29269', '29271', '29272', '29280', '29291', '29292', '29293', '29294', '29299', '30110', '30121', '30122', '30201', '30202', '30203', '30310', '30320', '30331', '30332', '30391', '30392', '30393', '30399', '31111', '31112', '31113', '31114', '31120', '31201', '31202', '31311', '31312', '31321', '31322', '31910', '31920', '31991', '31999', '32011', '32019', '32021', '32029', '32091', '32099', '33110', '33120', '33201', '33202', '33209', '33301', '33302', '33303', '33309', '33401', '33402', '33409', '33910', '33920', '33931', '33932', '33933', '33991', '33992', '33993', '33999', '35111', '35112', '35113', '35114', '35119', '35120', '35130', '35200', '35300', '36010', '36020', '37011', '37012', '37021', '37022', '38110', '38120', '38130', '38210', '38220', '38230', '38240', '38311', '38312', '38321', '38322', '39001', '39009', '41111', '41112', '41119', '41121', '41122', '41129', '41210', '41221', '41222', '41223', '41224', '41225', '41226', '41229', '42110', '42121', '42122', '42123', '42129', '42131', '42132', '42133', '42134', '42135', '42136', '42137', '42138', '42139', '42201', '42202', '42203', '42204', '42209', '42311', '42312', '42321', '42322', '42411', '42412', '42420', '42491', '42492', '42499', '42500', '58111', '58112', '58113', '58121', '58122', '58123', '58190', '58211', '58212', '58219', '58221', '58222', '59111', '59112', '59113', '59114', '59120', '59130', '59141', '59142', '59201', '59202', '60100', '60210', '60221', '60222', '60229', '61210', '61220', '61291', '61299', '62010', '62021', '62022', '62090']

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
            if len(line) != 3: continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
    
class Patent20Processor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["26121", "26129", "26323", "26329", "26521", "27112", "27192", "27199", "27212", "27302", "28111", "28119", "28121", "28421", "28519", "29132", "29172", "29271", "33303", "33993"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
    
class Patent20MCProcessor(object):
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return [['26', '27', '28', '29', '33'], ['1', '2', '3', '4', '5', '9'], ['0', '1', '2', '3', '7', '9'], ['1', '2', '3', '9']]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = list(line[2:])
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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
    
class NsmcProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[2]
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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


class PawsProcessor(object):
    """Processor for the PAWS data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            if text_a == "" or text_b == "":
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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


class KorSTSProcessor(object):
    """Processor for the KorSTS data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return [None]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[5]
            text_b = line[6]
            label = line[4]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file  # Only mnli for training
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, self.args.task, file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, self.args.task, file_to_read)), mode
        )


class QuestionPairProcessor(object):
    """Processor for the Question-Pair data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if text_a == "" or text_b == "":
                continue
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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


class HateSpeechProcessor(object):
    """Processor for the Korean Hate Speech data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["none", "offensive", "hate"]

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
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[3]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
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


# TODO
seq_cls_processors = {        
    "patent-en-all": PatentENAllProcessor,
    "patent-all-2": PatentAllProcessor2,
    "patent-all": PatentAllProcessor,
    "patent-20": Patent20Processor,
    "patent-20-mc": Patent20MCProcessor,
    "kornli": KorNLIProcessor,
    "nsmc": NsmcProcessor,
    "paws": PawsProcessor,
    "korsts": KorSTSProcessor,
    "question-pair": QuestionPairProcessor,
    "hate-speech": HateSpeechProcessor,
}

# TODO
seq_cls_tasks_num_labels = {"patent-en-all":176, "patent-all-2":499, "patent-all":563, "patent-20":20, "patent-20-mc":2, "kornli": 3, "nsmc": 2, "paws": 2, "korsts": 1, "question-pair": 2, "hate-speech": 3}

# TODO
seq_cls_output_modes = {
    "patent-en-all": "classification",
    "patent-all-2": "classification",
    "patent-all": "classification",
    "patent-20": "classification",
    "patent-20-mc": "classification",
    "kornli": "classification",
    "nsmc": "classification",
    "paws": "classification",
    "korsts": "regression",
    "question-pair": "classification",
    "hate-speech": "classification",
}


def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    processor = seq_cls_processors[args.task](args)
    output_mode = seq_cls_output_modes[args.task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task), list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        
        if args.multiclass:
            features = seq_cls_convert_examples_to_features_multiclass(
                args, examples, tokenizer, max_length=args.max_seq_len, task=args.task
            )
        else:
            features = seq_cls_convert_examples_to_features(
                args, examples, tokenizer, max_length=args.max_seq_len, task=args.task
            )
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
    temp_label_list = [l_idx for l_idx in range(args.num_labels)]
    if args.label_embedding == True:
        input_label_seq_tensor = torch.tensor([temp_label_list for _ in range(len(features))], dtype=torch.long)    
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, input_label_seq_tensor)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
    return dataset
