from .layer_init import init_linear_layer
from .prefix_encoder import PrefixModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, AutoTokenizer

class DualEncoderModel(PrefixModel):
    def __init__(self, config):
        super().__init__(config)
        self.st_projection_layer = nn.Linear(768, self.encoding_projection_size)
        self.st_projection_layer.apply(init_linear_layer)

        for param in self.prefix_encoder.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def average_fusion(self, encoding_0, encoding_1):
        return (encoding_0 + encoding_1)/2

    def forward(
        self,
        st_inputs,
        prefix_inputs,
        interview_lengths,
        labels=None,
    ):
        batch_size = len(interview_lengths)
        max_interview_length = max(interview_lengths)
        self.lstm.flatten_parameters()

        st_outputs = torch.zeros(
            size=(batch_size, max_interview_length, self.encoding_projection_size), 
            dtype=torch.float, 
            device=self.device
        )
        prefix_outputs = torch.zeros(
            size=(batch_size, max_interview_length, self.encoding_projection_size), 
            dtype=torch.float, 
            device=self.device
        )
        for sample_idx in range(batch_size): 
            input_ids = st_inputs[sample_idx][:interview_lengths[sample_idx],0]
            attention_mask = st_inputs[sample_idx][:interview_lengths[sample_idx],1]
            outputs = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            st_encodings = self.mean_pooling(outputs, attention_mask)
            st_encodings = F.normalize(st_encodings, p=2, dim=1) 
            st_encodings = self.dropout(st_encodings)
            st_encodings = self.st_projection_layer(st_encodings)
            st_outputs[sample_idx][:interview_lengths[sample_idx]] = st_encodings

            input_ids = prefix_inputs[sample_idx][:interview_lengths[sample_idx],0]
            attention_mask = prefix_inputs[sample_idx][:interview_lengths[sample_idx],1]
            outputs = self.prefix_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            prefix_encodings = torch.mean(outputs['hidden_states'][-1], dim=1)
            prefix_encodings = self.dropout(prefix_encodings)
            prefix_encodings = self.prefix_projection_layer(prefix_encodings)
            prefix_outputs[sample_idx][:interview_lengths[sample_idx]] = prefix_encodings

        encoder_outputs = self.average_fusion(st_outputs, prefix_outputs)

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

def init_dual_encoder(hparam):
    config = AutoConfig.from_pretrained(hparam.transformer_pretrained_id)

    config.transformer_pretrained_id = hparam.transformer_pretrained_id
    config.lstm_input_size = hparam.lstm_input_size
    config.lstm_hidden_size = hparam.lstm_hidden_size
    config.lstm_num_layers = hparam.lstm_num_layers
    config.dropout_prob = hparam.dropout_prob
    config.problem_type = hparam.problem_type
    config.num_labels = hparam.num_labels
    config.encoding_projection_size = hparam.encoding_projection_size
    config.pre_seq_len = hparam.pre_seq_len
    config.device = hparam.device

    model = DualEncoderModel(config=config)
    if len(hparam.device_id) > 1:
        model = nn.DataParallel(model)
    
    return model

class DualEncoderCollator():
    """
    Collate QR responses with ST and Roberta tokenizers, output ST and Roberta tokens, batch interview lengths, and labels
    """
    def __init__(self, transformer_pretrained_id, st_max_token_len, pre_seq_len, prefix_max_token_len):
        self.st_tokenizer = AutoTokenizer.from_pretrained(transformer_pretrained_id)
        self.st_max_token_len = st_max_token_len
        self.prefix_tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        self.pre_seq_len = pre_seq_len
        self.prefix_max_token_len = prefix_max_token_len - self.pre_seq_len

    def encode_sample(self, sample, tokenizer, max_token_len):
        encoded_sample = tokenizer(
            sample,
            padding='max_length',
            truncation=True,
            max_length=max_token_len,
            return_tensors='pt'
        )
        input_ids = encoded_sample['input_ids']
        attention_mask = encoded_sample['attention_mask']
        return torch.cat((input_ids.unsqueeze(1), attention_mask.unsqueeze(1)),dim=1)

    def __call__(self, batch):
        outputs = {}
        bs = len(batch)

        batch_sentences = [sample['text_input'] for sample in batch]
        batch_interview_length = torch.tensor([len(sentences) for sentences in batch_sentences])
        max_interview_length = max(batch_interview_length)

        batch_st_input = torch.zeros(size=(bs, max_interview_length, 2, self.st_max_token_len), dtype=torch.long)
        batch_booster_input = torch.zeros(size=(bs, max_interview_length, 2, self.prefix_max_token_len), dtype=torch.long)

        for sample_idx, sample in enumerate(batch_sentences):
            batch_st_input[sample_idx][:len(sample)] = self.encode_sample(sample, self.st_tokenizer, self.st_max_token_len)
            batch_booster_input[sample_idx][:len(sample)] = self.encode_sample(sample, self.prefix_tokenizer, self.prefix_max_token_len)
    
        outputs['st_inputs'] = batch_st_input
        outputs['prefix_inputs'] = batch_booster_input
        outputs['interview_lengths'] = batch_interview_length
        outputs['labels'] = torch.tensor([sample['score_label'] for sample in batch])

        return outputs

