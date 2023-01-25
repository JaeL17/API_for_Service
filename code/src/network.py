import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ElectraModel, BertModel, ElectraPreTrainedModel, BertPreTrainedModel,BartPretrainedModel
# from transformers.models.bart.modeling_bart import BartDecoder

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import get_activation

from transformers import BartConfig
from bart_network import BartDecoder


######################################################################################################
class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        # # Use GPU
        # if self.gpu:
        #     self.Q_proj = self.Q_proj.cuda()
        #     self.K_proj = self.K_proj.cuda()
        #     self.V_proj = self.V_proj.cuda()

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if not last_layer:
            outputs = torch.nn.functional.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        query_masks = query_masks.reshape([outputs.shape[0], outputs.shape[1], outputs.shape[2]])

        outputs = outputs * query_masks

        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            return outputs

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs
    
######################################################################################################


# ELECTRA + Label Embedding
class ElectraForSequenceClassification_exp1(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
                
        ####
        self.label_emb = nn.Embedding(config.num_labels, config.hidden_size)
        self.label_attn = multihead_attention(config.hidden_size, num_heads=1, dropout_rate=config.hidden_dropout_prob)
        ####
        self.lstm = nn.LSTM(config.hidden_size, int(config.hidden_size/2), bidirectional=True, batch_first=True)
        
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        input_label_seq_tensor=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0] # (batch_size, max_length, hidden_size)        
        
        label_embs = self.label_emb(input_label_seq_tensor) # [batch_size, label_nums, hidden_size]
        
        label_embs = sequence_output[:,0,:] * label_embs
                        
        word2label_att_outputs = self.label_attn(sequence_output, label_embs, label_embs, False) # (batch_size, max_length, hidden_size)
        
        lstm_out, _ = self.lstm(word2label_att_outputs)
        
        # cls_output = word2label_att_outputs[:, 0, :]
        
        logits = self.classifier(lstm_out)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


######################################################################################################

# ELECTRA + LSTM(num_labels iteration)
class ElectraForSequenceClassification_exp2(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.temp_num_labels
        self.config = config
        self.electra = ElectraModel(config)
        
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=False, batch_first=True)
        
        print('[***]', self.num_labels)
        
        self.out_layers = nn.ModuleList([nn.Linear( config.hidden_size, len(self.num_labels[i]) ) for i in range(len(self.num_labels))])
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        

        sequence_output = discriminator_hidden_states[0]
        cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
        
        cls_output = torch.unsqueeze(cls_output, 1)
        
        # self.num_classes 마다 num_labels 크기 반영
        total_loss = 0
        hidden = None
        output_logits = []        
        for i in range(len(self.num_labels)):
#             print('[-]', cls_output.shape)
#             if hidden is not None:
#                 print('[-]', hidden[0].shape)
            lstm_out, hidden = self.lstm(cls_output, hidden) # [batch_size, lstm_hidden_size]
            # print('[--]', lstm_out.shape, hidden[0].shape)
            logits = self.out_layers[i](lstm_out)
            
            output_logits.append(logits)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, len(self.num_labels[i])), labels[:, i].view(-1))
                total_loss += loss

        if not return_dict:
            # output = (logits,) + discriminator_hidden_states[1:]
            output = (output_logits,)            
            return ((total_loss,) + output) if total_loss is not None else output

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=output_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
    

######################################################################################################

class Decoder_LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=3, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, decoder_input=None, decoder_hidden_in=None, seq_len=1):
        decoder_emb = self.decoder_embed(decoder_input) # [batch_size, decoder_seqlen, hidden_size]        
        
        # if do_infer == True:
        if seq_len == 1:
            decoder_emb = torch.unsqueeze(decoder_emb, 1)
        
            # print('[*****]', decoder_emb.shape, decoder_hidden_in[0].shape)
        decoder_out, decoder_hidden_out = self.lstm(decoder_emb, decoder_hidden_in) # [batch_size, decoder_seqlen, hidden_size]
        
        # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        return decoder_out, decoder_hidden_out

# ELECTRA + Decoder(LSTM)
class ElectraForSequenceClassification_exp3(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.electra = ElectraModel(config)
        self.decoder = Decoder_LSTM(config)
        
        self.out_layers = nn.Linear( config.hidden_size, self.num_labels )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        decoder_input=None,
        decoder_hidden_in=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_decoding=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]        
        cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
                
        cls_output_temp = torch.unsqueeze(cls_output, 0) # [1, batch_size, hidden_size]        
        cls_output_temp = cls_output_temp.expand(3, -1, self.config.hidden_size) # [layer_num, batch_size, hidden_size]
        
        if labels == None and do_decoding:
            cls_output_temp = cls_output_temp.unsqueeze(2).expand(-1, -1, 5, -1)
            cls_output_temp = cls_output_temp.contiguous().view(3, -1, self.config.hidden_size) # [layer_num, batch_size, hidden_size]
            decoder_hidden_in = cls_output_temp.contiguous() # [batch_size, hidden_size]
            
            temp_decoder_input = decoder_input # [batch_size, 1]            
            temp_decoder_hidden = (decoder_hidden_in, decoder_hidden_in)
            _decoder_out, _decoder_hidden_out = self.decoder(temp_decoder_input, temp_decoder_hidden, seq_len=1)
            
            _logits = self.out_layers(_decoder_out) # [:,-1,:].squeeze(1) # [batch_size, num_labels]
            # _preds = torch.argmax(_logits, dim=1) # [batch_size, 1]
            
            return _logits
        
        if labels == None:
            list_pred = []
            list_logit = []
            
            ## BEAM ##
            cls_output_temp = cls_output_temp.unsqueeze(2).expand(-1, -1, 5, -1)
            cls_output_temp = cls_output_temp.contiguous().view(3, -1, self.config.hidden_size) # [layer_num, batch_size, hidden_size]
            ## BEAM ##
            
            decoder_hidden_in = cls_output_temp.contiguous() # [batch_size, hidden_size]
                        
            temp_decoder_input = decoder_input # [batch_size, 1]            
            temp_decoder_hidden = (decoder_hidden_in, decoder_hidden_in)
            for i in range(5):
                if i > 0:
                    # print(temp_decoder_input.shape)
                    # print('[**]', i, temp_decoder_input.shape, _preds.shape)
                    temp_decoder_input = torch.cat( (temp_decoder_input.view(-1, i), _preds.view(-1, 1)), dim=1 )
                # print('[*]', i, temp_decoder_input.shape, temp_decoder_hidden[0].shape)
                _decoder_out, _decoder_hidden_out = self.decoder(temp_decoder_input, temp_decoder_hidden, seq_len=i+1)
                # print('[****]', _decoder_out.shape)
                _logits = self.out_layers(_decoder_out)[:,i,:].squeeze(1) # [batch_size, num_labels]                
                
                # print('[*]', _logits.shape, _logits)
                _preds = torch.argmax(_logits, dim=1) # [batch_size, 1]                
                # print('[**]', _preds.shape, _preds)
                
                # temp_decoder_input = _preds
                # temp_decoder_hidden = _decoder_hidden_out         
                list_logit.append(torch.unsqueeze(_logits, 1))
                # list_pred.append(_preds)
            
            # preds = torch.cat(list_pred, 1)
            logits = torch.cat(list_logit, 1)
            
            return (None, logits, None)
        else:        
            # 학습 할 때(teacher-forcing) & 추론 처음 할 때
            if decoder_hidden_in == None:
                decoder_hidden_in = cls_output_temp.contiguous() # [layer_num, batch_size, hidden_size]
            decoder_out, decoder_hidden_out = self.decoder(decoder_input, (decoder_hidden_in, decoder_hidden_in), seq_len=5)

            logits = self.out_layers(decoder_out)
            
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # print('[*]', logits.shape, labels.shape, self.config.num_labels)
            # print('[**]', logits.view(-1, self.config.num_labels))
            # print('[***]', labels.view(-1))
            
            loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
            # print('[*]', loss)

        output = (logits, )
        return ((loss,) + output) # if loss is not None else output


# ######################################################################################################

class Decoder_LSTM_bakup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=3, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, decoder_input=None, decoder_hidden_in=None, seq_len=1):
        decoder_emb = self.decoder_embed(decoder_input) # [batch_size, decoder_seqlen, hidden_size]        
        
        # if do_infer == True:
        if seq_len == 1:
            decoder_emb = torch.unsqueeze(decoder_emb, 1)
            
        decoder_out, decoder_hidden_out = self.lstm(decoder_emb, decoder_hidden_in) # [batch_size, decoder_seqlen, hidden_size]
        
        # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        return decoder_out, decoder_hidden_out

# ELECTRA + Decoder(LSTM)
class ElectraForSequenceClassification_exp3_bakup(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.electra = ElectraModel(config)
        self.decoder = Decoder_LSTM_bakup(config)
        
        self.out_layers = nn.Linear( config.hidden_size, self.num_labels )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        decoder_input=None,
        decoder_hidden_in=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        infer_num=5,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]        
        cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
                
        cls_output_temp = torch.unsqueeze(cls_output, 0) # [1, batch_size, hidden_size]        
        cls_output_temp = cls_output_temp.expand(3, -1, self.config.hidden_size) # [layer_num, batch_size, hidden_size]
        
        if labels == None:
            list_pred = []
            list_logit = []
                        
            decoder_hidden_in = cls_output_temp.contiguous() # [batch_size, hidden_size]
                        
            temp_decoder_input = decoder_input # [batch_size, 1]            
            temp_decoder_hidden = (decoder_hidden_in, decoder_hidden_in)
            for i in range(infer_num):
                if i > 0:
                    # print(temp_decoder_input.shape)
                    # print('[**]', i, temp_decoder_input.shape, _preds.shape)
                    temp_decoder_input = torch.cat( (temp_decoder_input.view(-1, i), _preds.view(-1, 1)), dim=1 )
                # print('[*]', i, temp_decoder_input.shape, temp_decoder_hidden[0].shape)
                _decoder_out, _decoder_hidden_out = self.decoder(temp_decoder_input, temp_decoder_hidden, seq_len=i+1)
                # print('[****]', _decoder_out.shape)
                _logits = self.out_layers(_decoder_out)[:,i,:].squeeze(1) # [batch_size, num_labels]                
                
                # print('[*]', _logits.shape, _logits)
                _preds = torch.argmax(_logits, dim=1) # [batch_size, 1]                
                # print('[**]', _preds.shape, _preds)
                
                # temp_decoder_input = _preds
                # temp_decoder_hidden = _decoder_hidden_out         
                list_logit.append(torch.unsqueeze(_logits, 1))
                # list_pred.append(_preds)
            
            # preds = torch.cat(list_pred, 1)
            logits = torch.cat(list_logit, 1)
            
            return (None, logits, None)
        else:        
            # 학습 할 때(teacher-forcing) & 추론 처음 할 때
            if decoder_hidden_in == None:
                decoder_hidden_in = cls_output_temp.contiguous() # [layer_num, batch_size, hidden_size]
            decoder_out, decoder_hidden_out = self.decoder(decoder_input, (decoder_hidden_in, decoder_hidden_in), seq_len=5)

            logits = self.out_layers(decoder_out)
            
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # print('[*]', logits.shape, labels.shape, self.config.num_labels)
            # print('[**]', logits.view(-1, self.config.num_labels))
            # print('[***]', labels.view(-1))
            
            loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
            # print('[*]', loss)

        output = (logits, )
        return ((loss,) + output) # if loss is not None else output
    

######################################################################################################

class Encoder_exp3(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.electra = ElectraModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]


class Decoder_exp3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=config.layer_nums, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, input_ids=None, hidden_states=None):
        decoder_emb = self.embed(input_ids) # [batch_size, decoder_seqlen, hidden_size]        
        out, hidden_states = self.lstm(decoder_emb, hidden_states) # [batch_size, decoder_seqlen, hidden_size]
                
        return out, hidden_states # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]


class Classifier_exp3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder_exp3(config)
        self.decoder = Decoder_exp3(config)
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        
    def forward(
        self,
        # encoder
        encoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        beam_size=None,
        # decoder
        decoder_input_ids=None):
        
        # 인코더 출력 생성
        encoder_hidden_states = self.encoder(
            encoder_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # 디코더 입력 전 처리
        cls_output = encoder_hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
        cls_output = torch.unsqueeze(cls_output, 0) # [1, batch_size, hidden_size]
        cls_output = cls_output.expand(self.config.layer_nums, -1, self.config.hidden_size) # [layer_nums, batch_size, hidden_size]
        
        # 디코더 히든 스테이트 준비
        cls_output = cls_output.contiguous()
        if beam_size != None:
            cls_output = torch.unsqueeze(cls_output, 2).expand(-1, -1, beam_size, -1)
            cls_output = cls_output.reshape(self.config.layer_nums, -1, self.config.hidden_size)
        decoder_hidden_states = (cls_output, cls_output)
        
        # 디코더 출력 생성
         # [batch_size, decoder_seqlen]
        decoder_out, decoder_hidden_states = self.decoder(decoder_input_ids, decoder_hidden_states)
         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_out)
        
        # 로스 계산
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
        
        return {'loss':loss, 'logits':logits}


######################################################################################################


class Encoder_exp3_3(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.electra = ElectraModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]


class Decoder_exp3_3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=config.layer_nums, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, input_ids=None, hidden_states=None):
        decoder_emb = self.embed(input_ids) # [batch_size, decoder_seqlen, hidden_size]        
        out, hidden_states = self.lstm(decoder_emb, hidden_states) # [batch_size, decoder_seqlen, hidden_size]
                
        return out, hidden_states # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]


class Classifier_exp3_3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder_exp3_3(config)
        self.decoder = Decoder_exp3_3(config)
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        
    def forward(
        self,
        # encoder
        encoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        beam_size=None,
        # decoder
        decoder_input_ids=None):
        
        # 인코더 출력 생성
        encoder_hidden_states = self.encoder(
            encoder_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        # 디코더 입력 전 처리
        cls_output_first = encoder_hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
        cls_output = torch.unsqueeze(cls_output_first, 0) # [1, batch_size, hidden_size]
        cls_output = cls_output.expand(self.config.layer_nums, -1, self.config.hidden_size) # [layer_nums, batch_size, hidden_size]
        
        # 디코더 히든 스테이트 준비
        cls_output = cls_output.contiguous()
        if beam_size != None:
            cls_output = torch.unsqueeze(cls_output, 2).expand(-1, -1, beam_size, -1)
            cls_output = cls_output.reshape(self.config.layer_nums, -1, self.config.hidden_size)
        decoder_hidden_states = (cls_output, cls_output)
        
        # 디코더 출력 생성
         # [batch_size, decoder_seqlen]
        decoder_out, decoder_hidden_states = self.decoder(decoder_input_ids, decoder_hidden_states)
         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        
        _batch_size, _decoder_seqlen, _hidden_size = decoder_out.shape
#         cls_output_first = torch.unsqueeze(cls_output_first, 1)
#         decoder_out = torch.cat([cls_output_first.expand(_batch_size, _decoder_seqlen, self.config.hidden_size), decoder_out], dim=2)
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_out)
        
        # 로스 계산
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
        
        return {'loss':loss, 'logits':logits}

######################################################################################################


class Encoder_exp3_4(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.electra = ElectraModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]


class Decoder_exp3_4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size*2, config.hidden_size, num_layers=config.layer_nums, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, input_ids=None, hidden_states=None, input_feeds=None):
        decoder_emb = self.embed(input_ids) # [batch_size, decoder_seqlen, hidden_size]
        if input_feeds != None:
            decoder_emb = torch.cat([input_feeds, decoder_emb], dim=2)
        out, hidden_states = self.lstm(decoder_emb, hidden_states) # [batch_size, decoder_seqlen, hidden_size]
        
        return out, hidden_states # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]


class Classifier_exp3_4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder_exp3_4(config)
        self.decoder = Decoder_exp3_4(config)
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
    
    def encode(
        self,
        encoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        # 인코더 출력 생성
        encoder_out = self.encoder(
            encoder_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return encoder_out
        
    def decode(self, encoder_out, decoder_input_ids=None, decoder_feeds=None, decoder_hidden_states=None, beam_size=None):
        
        if decoder_hidden_states == None:        
            # 디코더 입력 전 처리
            cls_output_first = encoder_out[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
            cls_output = torch.unsqueeze(cls_output_first, 0) # [1, batch_size, hidden_size]
            cls_output = cls_output.expand(self.config.layer_nums, -1, self.config.hidden_size) # [layer_nums, batch_size, hidden_size]

            # 디코더 히든 스테이트 준비
            cls_output = cls_output.contiguous()
            if beam_size != None:
                cls_output = torch.unsqueeze(cls_output, 2).expand(-1, -1, beam_size, -1)
                cls_output = cls_output.reshape(self.config.layer_nums, -1, self.config.hidden_size)
            decoder_hidden_states = (cls_output, cls_output)
        
        # 디코더 출력 생성
         # [batch_size, decoder_seqlen]
        decoder_out, decoder_hidden_states = self.decoder(decoder_input_ids, decoder_hidden_states, decoder_feeds)
         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        
        # print('[*]', decoder_out.shape)
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_out)
        
        return logits, decoder_hidden_states, decoder_out
#################################################################################################
class Encoder_bert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.bert = BertModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None
    ):
#         ,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
        )
#             position_ids,
#             head_mask,
#             inputs_embeds,
#             output_attentions,
#             output_hidden_states,
#             return_dict,
#         )
        
        return discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]


class Decoder_bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size*2, config.hidden_size, num_layers=config.layer_nums, bidirectional=False, batch_first=True)
        self.config = config
    
    def forward(self, input_ids=None, hidden_states=None, input_feeds=None):
        decoder_emb = self.embed(input_ids) # [batch_size, decoder_seqlen, hidden_size]
        if input_feeds != None:
            decoder_emb = torch.cat([input_feeds, decoder_emb], dim=2)
        out, hidden_states = self.lstm(decoder_emb, hidden_states) # [batch_size, decoder_seqlen, hidden_size]
        
        return out, hidden_states # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]

class Classifier_bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder_bert(config)
        #self.encoder = nn.DataParallel(self.encoder, device_ids=[0, 1, 2], output_device=0)
        self.decoder = Decoder_bert(config)
        #self.decoder = nn.DataParallel(self.decoder, device_ids=[0, 1, 2], output_device=0)
        
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
    
    def encode(
        self,
        encoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,):
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None):
        
        # 인코더 출력 생성
        encoder_out = self.encoder(
            encoder_input_ids,
            attention_mask,
            token_type_ids,)
#             position_ids,
#             head_mask,
#             inputs_embeds,
#             output_attentions,
#             output_hidden_states,
#             return_dict,
#         )
        
        return encoder_out
        
    def decode(self, encoder_out, decoder_input_ids=None, decoder_feeds=None, decoder_hidden_states=None, beam_size=None):
        
        if decoder_hidden_states == None:        
            # 디코더 입력 전 처리
            cls_output_first = encoder_out[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
            cls_output = torch.unsqueeze(cls_output_first, 0) # [1, batch_size, hidden_size]
            cls_output = cls_output.expand(self.config.layer_nums, -1, self.config.hidden_size) # [layer_nums, batch_size, hidden_size]

            # 디코더 히든 스테이트 준비
            cls_output = cls_output.contiguous()
            if beam_size != None:
                cls_output = torch.unsqueeze(cls_output, 2).expand(-1, -1, beam_size, -1)
                cls_output = cls_output.reshape(self.config.layer_nums, -1, self.config.hidden_size)
            decoder_hidden_states = (cls_output, cls_output)
        
        # 디코더 출력 생성
         # [batch_size, decoder_seqlen]
        decoder_out, decoder_hidden_states = self.decoder(decoder_input_ids, decoder_hidden_states, decoder_feeds)
         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
        
        # print('[*]', decoder_out.shape)
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_out)
        
        return logits, decoder_hidden_states, decoder_out

######################################################################################################


class Encoder_exp4_0(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.electra = ElectraModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]

class Decoder_exp4_0(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        # self.embed = nn.Embedding(20, config.hidden_size)
        
        config.vocab_size = 20
        self.bart_decoder = BartDecoder(config)
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        encoder_outputs=None,
        encoder_attention_mask_temp=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
       ):
        # decoder_input_ids, [batch_size, decoder_seq_len]
        # decoder_attention_mask, [batch_size, decoder_seq_len]
        # encoder_hidden_states, [batch_size, encoder_seq_len, hidden_size]
        # encoder_attention_mask, [batch_size, encoder_seq_len, hidden_size]
        
#         print('[*]', decoder_input_ids.shape)
#         print('[**]', decoder_attention_mask.shape)
#         print('[***]', encoder_outputs.shape)
        decoder_outputs = self.bart_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask_temp,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # decoder_outputs.last_hidden_state,
        # decoder_outputs.past_key_values,
        # decoder_outputs.hidden_states,
        # decoder_outputs.attentions,
        # decoder_outputs.cross_attentions

        return decoder_outputs # [batch_size, decoder_seqlen, hidden_size]

class Classifier_exp4_0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder_exp4_0(config)
        
        json_path = '/docker/data/raw/base_model_2/bart_base_4.json'
        bart_config = BartConfig.from_pretrained(json_path)
        # BartConfig(json_path)
        # print('[__]', bart_config)
        self.decoder = Decoder_exp4_0(bart_config)
        
        # self.decoder = Decoder_exp4_0(config)
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
    
    def encode(
        self,
        encoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        # 인코더 출력 생성
        encoder_out = self.encoder(
            encoder_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return encoder_out
        
    def decode(self, encoder_out, decoder_input_ids=None, decoder_attention_mask=None, encoder_attention_mask=None):
        
        # 디코더 출력 생성
         # [batch_size, decoder_seqlen]
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_out,
            encoder_attention_mask_temp=encoder_attention_mask,
        )
        decoder_out = decoder_outputs.last_hidden_state
        
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_out)
        
        return logits
    
######################################################################################################

class Encoder_exp4(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.electra = ElectraModel(config)        
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return encoder_outputs


class Decoder_exp4(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.embed = nn.Embedding(config.num_labels, config.hidden_size)
        # self.embed = nn.Embedding(20, config.hidden_size)
        
        config.vocab_size = 20
        self.bart_decoder = BartDecoder(config)
        self.init_weights()
        self.config = config
    
    def forward(
        self,
        encoder_outputs=None,
        encoder_attention_mask_temp=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
       ):
        # decoder_input_ids, [batch_size, decoder_seq_len]
        # decoder_attention_mask, [batch_size, decoder_seq_len]
        # encoder_hidden_states, [batch_size, encoder_seq_len, hidden_size]
        # encoder_attention_mask, [batch_size, encoder_seq_len, hidden_size]
        
        decoder_outputs = self.bart_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask_temp,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # decoder_outputs.last_hidden_state,
        # decoder_outputs.past_key_values,
        # decoder_outputs.hidden_states,
        # decoder_outputs.attentions,
        # decoder_outputs.cross_attentions

        return decoder_outputs # [batch_size, decoder_seqlen, hidden_size]


class Classifier_exp4(nn.Module):
    def __init__(self, config, bart_config):
        super().__init__()
        self.encoder = Encoder_exp4(config)
        self.decoder = Decoder_exp4(bart_config)
        self.out_layers = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        
    def forward(
        self,
        # encoder
        encoder_input_ids=None,
        encoder_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        beam_size=None,
        # decoder
        decoder_input_ids=None,
        decoder_attention_mask=None):
        
        # 인코더 출력 생성
        encoder_outputs = self.encoder(
            encoder_input_ids,
            encoder_attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        # encoder_outputs # [batch_size, seq_len, hidden_size]
        
        # print(decoder_input_ids.shape, decoder_attention_mask.shape) 
        print(encoder_outputs[0][:,0,:].shape)
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs[0][:,0,:]
        )
        
        # 최종 출력 로짓 벡터 생성
        logits = self.out_layers(decoder_outputs[0])
        
        # 로스 계산
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
        
        return {'loss':loss, 'logits':logits}
    
######################################################################################################

# ELECTRA DECODER + concat(CLS_OUT, INPUT)

# class Decoder_LSTM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.decoder_embed = nn.Embedding(config.num_labels, config.hidden_size)
#         self.lstm = nn.LSTM(config.hidden_size*2, config.hidden_size*2, num_layers=3, bidirectional=False, batch_first=True)
#         self.config = config
    
#     def forward(self, decoder_input=None, decoder_hidden_in=None, cls_out=None, do_infer=False):
#         decoder_emb = self.decoder_embed(decoder_input) # [batch_size, decoder_seqlen, hidden_size]
        
#         if do_infer == True:
#             decoder_emb = torch.unsqueeze(decoder_emb, 1)
#             # cls_out = cls_out.expand(-1, 1, self.config.hidden_size)
#             decoder_emb = torch.cat((decoder_emb, cls_out), dim=2)
#         else:
#             # print('[*]', cls_out.shape)
#             cls_out = cls_out.expand(-1, 5, self.config.hidden_size)
#             decoder_emb = torch.cat((decoder_emb, cls_out), dim=2)
#         # print(decoder_emb.shape, decoder_hidden_in[0].shape)
#         decoder_out, decoder_hidden_out = self.lstm(decoder_emb, decoder_hidden_in) # [batch_size, decoder_seqlen, hidden_size]
        
#         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
#         return decoder_out, decoder_hidden_out

# # ELECTRA + Decoder(LSTM)
# class ElectraForSequenceClassification_exp3(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
        
#         self.electra = ElectraModel(config)
#         self.decoder = Decoder_LSTM(config)
        
#         self.out_layers = nn.Linear( config.hidden_size*2, self.num_labels )
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         decoder_input=None,
#         decoder_hidden_in=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
#             config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         discriminator_hidden_states = self.electra(
#             input_ids,
#             attention_mask,
#             token_type_ids,
#             position_ids,
#             head_mask,
#             inputs_embeds,
#             output_attentions,
#             output_hidden_states,
#             return_dict,
#         )

#         sequence_output = discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]        
#         cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
        
#         cls_output_input = torch.unsqueeze(cls_output, 1)
        
#         cls_output_temp = torch.unsqueeze(cls_output, 0) # [1, batch_size, hidden_size]        
#         cls_output_temp = cls_output_temp.expand(3, -1, self.config.hidden_size)
#         cls_output_temp = torch.cat((cls_output_temp, cls_output_temp), dim=2)
        
#         if labels == None:
#             list_pred = []
#             list_logit = []
            
#             decoder_hidden_in = cls_output_temp.contiguous() # [batch_size, hidden_size]
            
#             # print('[*]', decoder_input.reshape(-1, 1).shape)
                        
#             temp_decoder_input = decoder_input # [batch_size, 1]
#             # print('[*]', temp_decoder_input.shape)
#             temp_decoder_hidden = (decoder_hidden_in, decoder_hidden_in)
#             for i in range(5):
#                 _decoder_out, _decoder_hidden_out = self.decoder(temp_decoder_input, temp_decoder_hidden, cls_output_input, do_infer=True)            
#                 _logits = self.out_layers(_decoder_out).squeeze(1) # [batch_size, num_labels]                
                
#                 # print('[*]', _logits.shape, _logits)
#                 _preds = torch.argmax(_logits, dim=1) # [batch_size, 1]                
#                 # print('[**]', _preds.shape, _preds)
                
#                 temp_decoder_input = _preds
#                 temp_decoder_hidden = _decoder_hidden_out         
#                 list_logit.append(torch.unsqueeze(_logits, 1))
#                 # list_pred.append(_preds)
            
#             # preds = torch.cat(list_pred, 1)
#             logits = torch.cat(list_logit, 1)
            
#             return (None, logits, None)
#         else:        
#             # 학습 할 때(teacher-forcing) & 추론 처음 할 때
#             if decoder_hidden_in == None:
#                 decoder_hidden_in = cls_output_temp.contiguous() # [batch_size, hidden_size]
#             decoder_out, decoder_hidden_out = self.decoder(decoder_input, (decoder_hidden_in, decoder_hidden_in), cls_output_input)

#             logits = self.out_layers(decoder_out)
            
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # print('[*]', logits.shape, labels.shape, self.config.num_labels)
#             # print('[**]', logits.view(-1, self.config.num_labels))
#             # print('[***]', labels.view(-1))
            
#             loss = loss_fct(logits.contiguous().view(-1, self.config.num_labels), labels.contiguous().view(-1))
#             # print('[*]', loss)

#         output = (logits, )
#         return ((loss,) + output) # if loss is not None else output

    
# ######################################################################################################
# # ELECTRA + Decoder(Transformers)

# class Decoder_Transformer(ElectraPreTrainedModel):
#     def __init__(self, config):        
#         self.embeddings = ElectraEmbeddings(config)
#         self.encoder = ElectraEncoder(config)
    
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#         ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         batch_size, seq_length = input_shape
#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device)
#         if token_type_ids is None:
#             if hasattr(self.embeddings, "token_type_ids"):
#                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
#         hidden_states = self.embeddings(
#             input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
#         )
        
#         hidden_states = self.encoder(
#             hidden_states,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         # [batch_size, decoder_seqlen, hidden_size], [batch_size, hidden_size]
#         return decoder_out, decoder_hidden_out

# class ElectraForSequenceClassification_exp4(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.temp_num_labels
#         self.config = config
        
#         self.electra = ElectraModel(config)
#         self.decoder = Decoder_Transformer(config)
        
#         self.out_layers = nn.Linear( config.hidden_size, len(self.label_vocab_size) )
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         decoder_input=None,
#         decoder_hidden_in=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
#             config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         discriminator_hidden_states = self.electra(
#             input_ids,
#             attention_mask,
#             token_type_ids,
#             position_ids,
#             head_mask,
#             inputs_embeds,
#             output_attentions,
#             output_hidden_states,
#             return_dict,
#         )

#         sequence_output = discriminator_hidden_states[0] # [batch_size, seq_len, hidden_size]        
#         cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS]), [batch_size, hidden_size]
#         cls_output = torch.unsqueeze(cls_output, 1) # [batch_size, hidden_size]
        
#         # 학습 할 때(teacher-forcing) & 추론 처음 할 때
#         if decoder_hidden_in == None:
#             decoder_hidden_in = encoder_out # [batch_size, hidden_size]
#         decoder_out, decoder_hidden_out = self.decoder(decoder_input, decoder_hidden_in)
               
#         logits = self.out_layers(decoder_out)
            
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, len(self.config.label_vocab_size)), labels[:, i].view(-1))

#         output = (output_logits, decoder_hidden_out)            
#         return ((loss,) + output) if loss is not None else output
