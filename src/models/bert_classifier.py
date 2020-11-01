from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertConfig, BertPreTrainedModel
from torch import nn


@Model.register('bert_classifier')
class BertClassifierWrapper(Model):
    def __init__(self, vocab: Vocabulary,
                 num_layers=9,
                 hidden_size=200,
                 intermediate_size=200,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 num_attention_heads=3):
        super().__init__(vocab)
        config = BertConfig(
            vocab_size=vocab.get_vocab_size('tokens'),
            num_labels=vocab.get_vocab_size('labels'),
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=19,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            intermediate_size=intermediate_size
        )
        self.model = BertClassifier(config)
        self._accuracy = CategoricalAccuracy()
        self._op_split_accuracy = CategoricalAccuracy()
        self._len_split_accuracy = CategoricalAccuracy()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                answer: torch.IntTensor = None,
                dataset_type: torch.Tensor = None,
                metadata: Dict = None) -> Dict[str, torch.Tensor]:
        input_ids = tokens['tokens']
        loss, logits = self.model(input_ids, labels=answer)
        outputs = {'loss': loss, 'logits': logits}

        if answer is not None:
            self._update_metrics(logits, answer, dataset_type)

        return outputs

    def _update_metrics(self, logits, answer, dataset_type):
        self._accuracy(logits, answer, dataset_type == 0)
        self._op_split_accuracy(logits, answer, dataset_type == 1)
        self._len_split_accuracy(logits, answer, dataset_type == 2)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "op_split_accuracy": self._op_split_accuracy.get_metric(reset),
            "len_split_accuracy": self._len_split_accuracy.get_metric(reset)
        }
        return metrics


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

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
    ):

        outputs = self.bert(
            input_ids['tokens'],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        org_output = outputs[0][:,0]

        logits = self.classifier(org_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
