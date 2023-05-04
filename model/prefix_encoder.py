import torch.nn as nn
import torch
from .layer_init import init_linear_layer, init_rnn
from transformers import AutoModel, AutoConfig, RobertaModel, RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .attention import AttentionLayer

class PrefixEncoder(torch.nn.Module):
    """Prefix implementation modified from https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py """
    def __init__(self, pre_seq_len, num_hidden_layers, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix_tokens):
        return self.embedding(prefix_tokens)  

class RobertaPrefixEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.init_weights()

        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.pre_seq_len, self.n_layer, config.hidden_size)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))

    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

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

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class InterviewBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_pretrained_id = config.transformer_pretrained_id
        self.device = config.device
        self.num_labels = config.num_labels

        self.transformer_encoder = AutoModel.from_pretrained(self.transformer_pretrained_id, config=config)
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

        self.lstm = torch.nn.LSTM(
            input_size=config.lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout_prob,
            batch_first=True,
            bidirectional=True
        )
        init_rnn(self.lstm)

        self.attention = AttentionLayer(
            device=self.device,
            hidden_size=config.lstm_hidden_size,
            bidirectional=True
        )

        self.dropout = torch.nn.Dropout(config.dropout_prob)
        self.prediction_head = torch.nn.Linear(config.lstm_hidden_size*2, config.num_labels)
        self.prediction_head.apply(init_linear_layer)

class PrefixModel(InterviewBaseModel):
    def __init__(self, config):
        super().__init__(config)
        prefix_config = AutoConfig.from_pretrained('roberta-base')
        prefix_config.pre_seq_len = config.pre_seq_len
        self.prefix_encoder = RobertaPrefixEncoder.from_pretrained('roberta-base', config=prefix_config)

        self.encoding_projection_size = config.encoding_projection_size
        self.prefix_projection_layer = nn.Linear(768, config.encoding_projection_size)

    def forward(
        self,
        inputs,
        interview_lengths,
        labels=None,
    ):
        batch_size = len(interview_lengths)
        max_interview_length = max(interview_lengths)
        self.lstm.flatten_parameters()

        encoder_outputs = torch.zeros(
            size=(batch_size, max_interview_length, self.encoding_projection_size), 
            dtype=torch.float, 
            device=self.device
        )
        for sample_idx in range(batch_size): 
            input_ids = inputs[sample_idx][:interview_lengths[sample_idx],0]
            attention_mask = inputs[sample_idx][:interview_lengths[sample_idx],1]
            outputs = self.prefix_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            prefix_encodings = torch.mean(outputs['hidden_states'][-1], dim=1) 
            prefix_encodings = self.dropout(prefix_encodings)
            prefix_encodings = self.prefix_projection_layer(prefix_encodings)
            encoder_outputs[sample_idx][:interview_lengths[sample_idx]] = prefix_encodings
        
        
        encoder_outputs = nn.utils.rnn.pack_padded_sequence(encoder_outputs, lengths=interview_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, (h_n, _) = self.lstm(encoder_outputs)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True) 
        attention_output, _ = self.attention(lstm_output, interview_lengths)
        attention_output = self.dropout(attention_output)
        logits = self.prediction_head(attention_output)
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )