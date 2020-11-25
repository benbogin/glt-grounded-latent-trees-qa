from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertConfig

from src.models.util.nn import tie_layers
from src.models.util.tree import sentence_to_tree
from src.models.glt import GLTEmbeddings, GLTLayer, GLTAnswerVisualTextComp


@Model.register('glt_ungrounded', exist_ok=True)
class CKYUngroundedWrapper(Model):
    def __init__(self, vocab: Vocabulary,
                 max_sentence_length: int,
                 hidden_size: int,
                 use_position_embeddings: bool,
                 hidden_dropout_prob: float,
                 answer_comp_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 tie_layer_norm: bool,
                 answer_pooler: bool,
                 input_img_dim: int = None,
                 layers_to_tie: dict = None,
                 layer_dropout_prob: float = 0.25
                 ):
        super().__init__(vocab)
        config = BertConfig(
            grounded=False,
            vocab_size=vocab.get_vocab_size('tokens'),
            num_labels=vocab.get_vocab_size('labels'),
            max_sentence_length=max_sentence_length,
            hidden_size=hidden_size,
            input_img_dim=input_img_dim,
            max_position_embeddings=max_sentence_length,
            use_position_embeddings=use_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_dropout_prob=layer_dropout_prob,
            answer_comp_dropout_prob=answer_comp_dropout_prob,
            intermediate_size=hidden_size,
            layers_to_tie=layers_to_tie,
            tie_layer_norm=tie_layer_norm,
            answer_pooler=answer_pooler,
            non_compositional_reps=False,
            visual_module_dropout_prob=0
        )
        self.model = CKYClassifier(config)
        self._accuracy = CategoricalAccuracy()
        self._op_split_accuracy = CategoricalAccuracy()
        self._len_split_accuracy = CategoricalAccuracy()

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                answer: torch.IntTensor = None,
                dataset_type: torch.Tensor = None,
                metadata: Dict = None) -> Dict[str, torch.Tensor]:
        outputs = self.model(tokens, answer=answer)

        loss, logits, all_states, attention_probs, debug_info = outputs

        if loss is None:
            # loss = None
            loss = torch.zeros(1)
        else:
            # do not compute non-clevr instances for loss
            loss = loss.mean()

        outputs_dict = {'loss': loss, 'logits': logits}

        if not self.training:
            # copy to cpu since calculations are done on cpu anyway
            attention_probs = [probs.cpu() if probs is not None else probs for probs in attention_probs]

            outputs_dict['tree_str'] = [sentence_to_tree(m['question'], attention_probs, batch_index) for batch_index, m
                                   in enumerate(metadata)]
            outputs_dict['question'] = [m['question'] for m in metadata]
            outputs_dict['answer'] = [m['answer'] for m in metadata]
            outputs_dict['dataset_type'] = dataset_type

            predictions = logits.argmax(dim=1).tolist()
            label_index_to_text_dict = self.vocab.get_index_to_token_vocabulary('labels')
            outputs_dict['prediction'] = [label_index_to_text_dict[idx] for idx in predictions]

            outputs_dict['correct'] = [p == a for p, a in zip(outputs_dict['prediction'], outputs_dict['answer'])]

            outputs_dict.update({k: v for k, v in debug_info.items() if v is not None})

        if answer is not None:
            self._update_metrics(logits, answer, dataset_type)

        return outputs_dict

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


class CKYClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(CKYClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.model = GroundedCKYModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        clsfr_input_size = config.hidden_size
        self.classifier = nn.Linear(clsfr_input_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            answer=None,
            pad_start=None,
    ):
        outputs = self.model(
            input_ids['tokens']['tokens'],
            pad_start=pad_start
        )

        final_answer = outputs[0]

        logits = self.classifier(final_answer)

        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here

        loss = None
        if answer is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), answer.view(-1))

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class GroundedCKYPooler(nn.Module):
    def __init__(self, config, input_dim=None):
        super(GroundedCKYPooler, self).__init__()
        if not input_dim:
            input_dim = config.hidden_size
        self.dense = nn.Linear(input_dim, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GroundedCKYModel(BertPreTrainedModel):
    def __init__(self, config):
        super(GroundedCKYModel, self).__init__(config)
        self.config = config

        self.embeddings = GLTEmbeddings(config)
        self.encoder = GroundedCKYEncoder(config)

        self.init_weights()

    def forward(self, input_ids=None, pad_start=None):
        embedding_output = self.embeddings(input_ids=input_ids)
        attention_mask = input_ids > 0

        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, pad_start=pad_start)

        return encoder_outputs


class GroundedCKYEncoder(nn.Module):
    def __init__(self, config):
        super(GroundedCKYEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.word_energy_lin = nn.Linear(config.hidden_size, 1)

        modules = [GLTLayer(config) for _ in range(config.max_sentence_length - 1)]

        layers_to_tie = config.layers_to_tie
        if layers_to_tie:
            tie_layers(modules[0], modules[1:], config.layers_to_tie)
        if config.tie_layer_norm:
            tie_layers(modules[13], modules[14:], ['pair_compose.output.LayerNorm'])
            tie_layers(modules[13], modules[14:], ['pair_compose.weighted_output.LayerNorm'])

        self.layer = nn.ModuleList(modules)
        self.answer_nn = GLTAnswerVisualTextComp(config)

    def forward(self, hidden_states, attention_mask, pad_start=None):
        all_hidden_states = [hidden_states]
        all_attentions = (None,)
        debug_info = {}

        batch_size, seq_length, dim = hidden_states.size()

        num_layers = min(len(self.layer), seq_length - 1)
        root_answers = torch.zeros(batch_size, dim).float().to(hidden_states.device)

        sentence_lengths = util.get_lengths_from_binary_sequence_mask(attention_mask)

        for i, layer_module in enumerate(self.layer[:num_layers]):
            layer_outputs = layer_module(all_hidden_states, i)
            hidden_states, attentions, layer_debug_info = layer_outputs

            # Add last layer
            all_hidden_states += [hidden_states]
            all_attentions = all_attentions + (attentions,)

            # we should only update the root hidden states when sentence is shorter than num_layers
            # so we keep a mask of the relevant batch indices
            update_root_indices_mask = (sentence_lengths - 2 >= i).float()
            update_root_indices_mask = update_root_indices_mask.unsqueeze(1).to(hidden_states.device)

            answer, answer_debug_info = self.answer_nn(hidden_states[:, 0])
            layer_debug_info.update(answer_debug_info)

            root_answers = root_answers * (1 - update_root_indices_mask) + update_root_indices_mask * answer

            for key, val in layer_debug_info.items():
                debug_info[f'{key}_{i+1}'] = val

        outputs = (root_answers, all_hidden_states, all_attentions, debug_info)
        return outputs  # last-layer hidden state, all hidden states, all attentions

