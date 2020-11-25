from collections import defaultdict
from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Average
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertConfig
from transformers.modeling_bert import BertIntermediate, BertLayerNorm, BertLayer

from src.models.util.nn import tie_layers
from src.models.util.tree import sentence_to_tree


@Model.register('glt', exist_ok=True)
class GLTWrapper(Model):
    def __init__(self, vocab: Vocabulary,
                 max_sentence_length: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 layer_dropout_prob: float,
                 answer_comp_dropout_prob: float,
                 transformer_dropout_prob: float,
                 classifier_dropout_prob: float,
                 visual_module_dropout_prob: float,
                 use_position_embeddings: bool = False,
                 input_img_dim: int = None,
                 visual_self_attention_layers: int = None,
                 layers_to_tie: dict = None,
                 glove_path: str = None,
                 non_compositional_reps: bool = False,
                 contextualize_inputs_layers: int = 0,
                 control_gate_add_skip: bool = True,
                 control_gate_add_intersect: bool = True,
                 control_gate_add_union: bool = True,
                 control_gate_add_vis: bool = True,
                 control_gate_set_vis_left_branching: bool = False,
                 control_gate_add_extra_vis_module_left_branching: bool = False,
                 answer_pooler: bool = False,
                 ):
        """
        :param max_sentence_length: the maximum number of sentence tokens, which dictates the number of layers
        :param hidden_size: hidden size of all representations ($h_{dim} in the paper)
        :param num_attention_heads: number of attention heads for transformer layers - this only affects the transformer
        layers performed on the visual features, and the contextualized embeddings experiment
        :param layer_dropout_prob: dropout probability to be used across different parts of the model
        :param answer_comp_dropout_prob: dropout probability to be used in the answer component
        :param transformer_dropout_prob: dropout probability to be used in transformer layer
        :param classifier_dropout_prob: dropout probability to be used in the classification layer (last layer)
        :param visual_module_dropout_prob: dropout probability to be used in the visual module
        :param use_position_embeddings: indicates whether position embeddings should be added
        :param input_img_dim: dimension of the image features
        :param visual_self_attention_layers: number of self-attention layers to be performed on the visual features
        :param layers_to_tie: list of the layer names to be tied with same parameters
        :param glove_path: path to glove embeddings file
        :param non_compositional_reps: boolean to indicate if non-compositional representations should be used (biLSTM)
        :param contextualize_inputs_layers: number of self-attention layers to be performed on input tokens
        :param control_gate_add_skip: indicates if skip module should be used
        :param control_gate_add_intersect: indicates if intersect module should be used
        :param control_gate_add_union: indicates if union module should be used
        :param control_gate_add_vis: indicates if visual module should be used
        :param control_gate_set_vis_left_branching: indicates if visual module should use left-branching (see paper)
        :param control_gate_add_extra_vis_module_left_branching: indicates if two visual modules should be used,
        one with left-branching and one with right-branching
        """
        super().__init__(vocab)
        config = BertConfig(
            grounded=True,
            vocab_size=vocab.get_vocab_size('tokens'),
            num_labels=vocab.get_vocab_size('labels'),
            max_sentence_length=max_sentence_length,
            hidden_size=hidden_size,
            input_img_dim=input_img_dim,
            visual_self_attention_layers=visual_self_attention_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_sentence_length,
            use_position_embeddings=use_position_embeddings,
            visual_module_dropout_prob=visual_module_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            layer_dropout_prob=layer_dropout_prob,
            hidden_dropout_prob=transformer_dropout_prob,
            answer_comp_dropout_prob=answer_comp_dropout_prob,
            attention_probs_dropout_prob=0,
            intermediate_size=hidden_size,
            layers_to_tie=layers_to_tie,
            glove_path=glove_path,
            non_compositional_reps=non_compositional_reps,
            contextualize_inputs_layers=contextualize_inputs_layers,
            control_gate_add_skip=control_gate_add_skip,
            control_gate_add_intersect=control_gate_add_intersect,
            control_gate_add_union=control_gate_add_union,
            control_gate_add_vis=control_gate_add_vis,
            control_gate_set_vis_left_branching=control_gate_set_vis_left_branching,
            control_gate_add_extra_vis_module_left_branching=control_gate_add_extra_vis_module_left_branching,
            answer_pooler=answer_pooler
        )
        self.model = GLTClassifier(config, vocab)
        
        # accuracy for in-distribution examples (CLEVR)
        self._accuracy = CategoricalAccuracy()
        
        # accuracy for out-of-distribution examples (CLOSURE)
        self._closure_accuracy = CategoricalAccuracy()
        
        # accuracy for out-of-distribution examples (CLOSURE), metric counting alternative answers because of
        # ambiguity issues (see paper)
        self._closure_accuracy_alternative = CategoricalAccuracy()

        # map family index of question to a name (for metrics grouped by groups)
        self._question_families = {
            'count_simple': {84},
            'query_simple': {85, 86, 87, 88, 89},
            'query_same_size': {52, 53, 54},
            'exist_same_size': {36, 44},
            'count_same_size': {40, 48},
            'query_same_color': {55, 56, 57},
            'exist_same_color': {37, 45},
            'count_same_color': {41, 49},
            'query_same_material': {58, 59, 60},
            'exist_same_material': {38, 46},
            'count_same_material': {42, 50},
            'query_same_shape': {61, 62, 63},
            'exist_same_shape': {39, 47},
            'count_same_shape': {43, 51},
            'count_either': {64, 65, 66, 68, 69},
            'query_pos': {74, 75, 76, 77},
            'count_pos_either': {70, 71},
            'count_pos_either_2': {67},
            'count_pos': {72},
            'exist_pos': {73},
            'count_pos_2steps': {78, 25},
            'exist_pos_2steps': {79, 26},
            'query_pos_2steps': {80, 81, 82, 83, 27, 28, 29, 30},
            'count_pos_both': {31},
            'query_pos_both': {32, 33, 34, 35},
            'compare_count': {0, 1, 2},
            'compare_count_pos': {3, 4, 5},
            'compare_count_pos_both': {6, 7, 8},
            'compare_attr_size': {9},
            'compare_attr_color': {10},
            'compare_attr_material': {11},
            'compare_attr_shape': {12},
            'compare_attr_pos_size': {13, 14, 15},
            'compare_attr_pos_color': {16, 17, 18},
            'compare_attr_pos_material': {19, 20, 21},
            'compare_attr_pos_shape': {22, 23, 24},
        }
        self._question_families_rev = {f: k for k, v in self._question_families.items() for f in v}

        self._accuracies_per_family = {k: Average() for k in self._question_families.keys()}
        self._accuracies_per_family['other'] = Average()
        self._accuracies_per_closure_group = defaultdict(Average)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                tokens_sent2: Dict[str, torch.LongTensor] = None,
                img_features: torch.Tensor = None,
                n_img_features: torch.Tensor = None,
                answer: torch.IntTensor = None,
                alternative_answer: torch.IntTensor = None,
                patches_pos: torch.Tensor = None,
                metadata: Dict = None,
                is_ood_example: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        :param tokens: dictionary holding first sentence tokens
        :param tokens_sent2: optional dictionary holding second sentence tokens (used when sentences with a semicolon
        are split; see paper)
        :param img_features: tensor of size N X O X D holding visual features (N=batch size, O=number of objects,
        D=visual feature dimension)
        :param n_img_features: vector of size N holding the predicted number of objects in the image
        (since img_features is padded) 
        :param answer: vector of size N holding the index of the answer for each example
        :param alternative_answer: vector of size N holding the index of the alternative answer for each example (alternative
        due to ambiguity issue, see paper)
        :param patches_pos: tensor of size N X O X 6 holding spatial position info of each object
        """

        input_ids = tokens['tokens']
        input2_ids = tokens_sent2['tokens'] if tokens_sent2 is not None else None
        outputs = self.model(input_ids, input2_ids, answer=answer,
                             img_features=img_features, n_img_features=n_img_features, patches_pos=patches_pos)

        loss, logits, final_den, sent1, sent2 = outputs
        all_states, attention_probs, _, _, all_den, debug_info = sent1

        if sent2:
            all_states2, attention_probs2, _, _, all_den2, debug_info2 = sent2

        if all(is_ood_example > 0) or loss is None:
            loss = torch.zeros(1)
        else:
            # do not compute non-clevr instances for loss
            loss = (loss * (is_ood_example == 0)).sum() / (is_ood_example == 0).sum()

        outputs_dict = {'loss': loss, 'logits': logits}

        if not self.training:
            [outputs_dict.update({'attention_probs_' + str(i + 1): p}) for i, p in enumerate(attention_probs[1:])]

            # copy to cpu since calculations are done on cpu anyway
            attention_probs = [probs.cpu() if probs is not None else probs for probs in attention_probs]

            outputs_dict['tree_str'] = [sentence_to_tree(m['question_tokens'], attention_probs, batch_index) for
                                        batch_index, m
                                        in enumerate(metadata)]
            outputs_dict['question'] = [m['question'] for m in metadata]
            outputs_dict['question_index'] = [m['question_index'] for m in metadata]
            outputs_dict['question_tokens'] = [m['question_tokens'] for m in metadata]
            outputs_dict['answer'] = [m['answer'] for m in metadata]
            outputs_dict['image_filename'] = [m['image_filename'] for m in metadata]
            outputs_dict['object_pos'] = [m['pos_info'] for m in metadata]
            outputs_dict['qst_family'] = [m['question_family_index'] for m in metadata]
            outputs_dict['closure_group'] = [m['closure_group'] for m in metadata]
            outputs_dict['alternative_answer'] = [m['alternative_answer'] for m in metadata]

            predictions = logits.argmax(dim=1).tolist()
            label_index_to_text_dict = self.vocab.get_index_to_token_vocabulary('labels')
            outputs_dict['prediction'] = [label_index_to_text_dict[idx] for idx in predictions]
            outputs_dict['is_ood_example'] = is_ood_example

            outputs_dict['correct'] = [p == a or p == a2 for p, a, a2 in
                                       zip(outputs_dict['prediction'], outputs_dict['answer'],
                                           outputs_dict['alternative_answer'])]

            if img_features is not None:
                outputs_dict['denotation'] = final_den
                [outputs_dict.update({'denotations_' + str(i + 1): p}) for i, p in enumerate(all_den)]
            outputs_dict.update({k: v for k, v in debug_info.items() if v is not None})

            if sent2:
                outputs_dict['question_tokens_sent2'] = [m['question_tokens_sent2'] for m in metadata]
                [outputs_dict.update({'attention_probs2_' + str(i + 1): p}) for i, p in enumerate(attention_probs2[1:])]
                [outputs_dict.update({'denotations2_' + str(i + 1): p}) for i, p in enumerate(all_den2)]
                outputs_dict.update({k + '_2': v for k, v in debug_info2.items()})
                outputs_dict['coref_gate'] = debug_info2['coref_gate']

        if answer is not None:
            self._update_metrics(logits, answer, alternative_answer,
                                 is_ood_example, metadata)

        return outputs_dict

    def _update_metrics(self, logits, answer, alternative_answer,
                        is_ood_example, metadata):
        self._accuracy(logits, answer, is_ood_example == 0)
        self._closure_accuracy(logits, answer, is_ood_example == 1)
        self._closure_accuracy_alternative(logits, alternative_answer, is_ood_example == 1)

        if not self.training:
            predictions = logits.max(dim=-1)[1]
            for i, q in enumerate(metadata):
                correct = (predictions[i] == answer[i]).item()
                question_family_index = metadata[i]['question_family_index']
                closure = metadata[i]['closure_group']

                if closure:
                    group_metric = self._accuracies_per_closure_group[closure]
                    group_metric(correct)
                    if 'embed_mat_spa' in closure:
                        correct_alt = (predictions[i] == alternative_answer[i]).item()
                        group_metric = self._accuracies_per_closure_group['embed_mat_spa_alt']
                        group_metric(correct_alt)
                else:
                    if question_family_index in self._question_families_rev:
                        family_name = self._question_families_rev[question_family_index]
                    else:
                        family_name = 'other'
                    family_metric = self._accuracies_per_family[family_name]

                    family_metric(correct)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "gen_accuracy": self._closure_accuracy.get_metric(reset),
            "_gen_accuracy_alt": self._closure_accuracy_alternative.get_metric(reset)
        }

        metrics.update({'_' + k: v.get_metric(reset) for k, v in self._accuracies_per_family.items() if v._count > 0})

        for k, acc_metric in self._accuracies_per_closure_group.items():
            if acc_metric._count == 0:
                continue
            metric_name = '_closure_' + k
            metrics[metric_name] = acc_metric.get_metric(reset)
        return metrics


class GLTClassifier(BertPreTrainedModel):
    def __init__(self, config, vocab: Vocabulary = None):
        super(GLTClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.model = GLTModel(config, vocab)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input2_ids=None,
            answer=None,
            img_features: torch.Tensor = None,
            n_img_features: torch.Tensor = None,
            patches_pos: torch.Tensor = None,
    ):
        outputs = self.model(
            input_ids['tokens'],
            input2_ids['tokens'],
            img_features=img_features,
            n_img_features=n_img_features,
            patches_pos=patches_pos
        )

        final_answer, final_den, sent1_outputs, sent2_outputs = outputs

        classifier_input = final_answer
        logits = self.classifier(classifier_input)

        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here

        loss = None
        if answer is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), answer.view(-1))
        outputs = (loss,) + outputs

        return outputs


class GLTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
       Same as BertEmbeddings except position embeddings can be disabled
    """

    def __init__(self, config, vocab: Vocabulary = None):
        super(GLTEmbeddings, self).__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_index=0,
                                         vocab_namespace='tokens')

        if hasattr(config, 'glove_path') and config.glove_path:
            assert vocab is not None
            self.word_embeddings_glove = Embedding.from_vocab_or_file(vocab, 300,
                                                                      pretrained_file=config.glove_path,
                                                                      projection_dim=config.hidden_size,
                                                                      trainable=False)
            self.word_embeddings_glove._pretrained_file = config.glove_path
            self.use_glove = bool(config.glove_path)
        else:
            self.use_glove = False

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.layer_dropout_prob)
        self.use_position_embeddings = config.use_position_embeddings

    def combine_embeddings(self, input_ids):
        """
        This is a kind-of hacky way to combine glove embeddings with the original learned embeddings when fine-tuning
        """
        original_embeddings_size = 73
        new_emb_mask = (input_ids >= original_embeddings_size).float().unsqueeze(-1)

        return self.word_embeddings(input_ids) * (1 - new_emb_mask) + self.word_embeddings_glove(
            input_ids) * new_emb_mask

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            if self.use_position_embeddings:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
            else:
                position_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            if self.use_glove:
                inputs_embeds = self.combine_embeddings(input_ids)
            else:
                inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds
        if self.use_position_embeddings:
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GLTPooler(nn.Module):
    def __init__(self, config, input_dim=None):
        super(GLTPooler, self).__init__()
        if not input_dim:
            input_dim = config.hidden_size
        self.dense = nn.Linear(input_dim, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GLTModel(BertPreTrainedModel):
    def __init__(self, config, vocab: Vocabulary):
        super(GLTModel, self).__init__(config)
        self.config = config

        self.embeddings = GLTEmbeddings(config, vocab)
        self.encoder = GLTEncoder(config)

        self.init_weights()

    def forward(self, input_ids=None, input2_ids=None, img_features: torch.Tensor = None,
                n_img_features: torch.Tensor = None, patches_pos: torch.Tensor = None):
        embedding_output = self.embeddings(input_ids=input_ids)
        attention_mask = input_ids > 0

        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask,
                                       img_features=img_features,
                                       n_img_features=n_img_features, patches_pos=patches_pos)

        (root_answers, all_hidden_states, all_attentions, vis_root_den, vis_emb,
         all_denotations, debug_info) = encoder_outputs

        if input2_ids.size(1) > 0:
            embedding_output2 = self.embeddings(input_ids=input2_ids)
            attention_mask2 = input2_ids > 0
            encoder_outputs_2 = self.encoder(embedding_output2, attention_mask=attention_mask2,
                                             img_features=img_features,
                                             n_img_features=n_img_features, patches_pos=patches_pos,
                                             prev_sent_den=vis_root_den)

            (root_answers2, all_hidden_states2, all_attentions2, vis_root_den2, vis_emb2,
             all_denotations2, debug_info2) = encoder_outputs_2

            # instances where we have two sentences
            root_answers2 = encoder_outputs_2[0]
            text2_exists_mask = (input2_ids.sum(dim=1) > 0).unsqueeze(-1).long()

            final_answers = text2_exists_mask * root_answers2 + (1 - text2_exists_mask) * root_answers
            final_den = text2_exists_mask * vis_root_den2 + (1 - text2_exists_mask) * vis_root_den
            return final_answers, final_den, encoder_outputs[1:], encoder_outputs_2[1:]
        else:
            return root_answers, vis_root_den, encoder_outputs[1:], None


class GLTTokenGrounding(nn.Module):
    """
    Grounds tokens to objects
    """
    def __init__(self, config):
        super(GLTTokenGrounding, self).__init__()
        self.initial_img_project = nn.Linear(config.input_img_dim, config.hidden_size)
        self.text_project = nn.Linear(config.hidden_size, config.hidden_size)

        self.b_bias = nn.Parameter(torch.zeros((1)))

        self.lnorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pos_project = nn.Linear(6, config.hidden_size)
        self.lnorm_pos = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.coreference_jump_gate = nn.Linear(config.hidden_size, 2)

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.visual_self_attention_layers)]
        )

    def forward(self, text_features, img_features: torch.Tensor = None,
                patches_pos: torch.Tensor = None, obj_mask: torch.Tensor = None, prev_sent_den=None):
        img_features_proj = self.initial_img_project(img_features)
        text_features_proj = self.text_project(text_features)

        pos_features = self.pos_project(patches_pos)
        img_features_proj += pos_features

        img_features_proj = self.lnorm(img_features_proj)

        for layer_module in self.layer:
            img_features_proj, = layer_module(img_features_proj, obj_mask.unsqueeze(1).unsqueeze(-1))

        batch_size, n_objects, d = img_features_proj.size()
        seq_len = text_features_proj.size(1)

        img_features_proj_exp = img_features_proj.view(batch_size, 1, n_objects, 1, d).expand(-1, seq_len, -1, -1, -1)
        text_features_proj_exp = text_features_proj.view(batch_size, seq_len, 1, d, 1).expand(-1, -1, n_objects, -1, -1)

        denotation_logits = (img_features_proj_exp.matmul(text_features_proj_exp).squeeze(-1).squeeze(-1) / (
                    d ** 0.5)) + self.b_bias
        masked_logits = (denotation_logits - (1 - obj_mask.unsqueeze(1)) * 1e6)

        denotation = torch.sigmoid(masked_logits)

        coref_gate = None
        if prev_sent_den is not None:
            coref_gate = self.coreference_jump_gate(text_features).softmax(dim=-1)
            prev_sent_den = prev_sent_den.unsqueeze(1).expand(-1, seq_len, -1)
            candidates = torch.stack((denotation, prev_sent_den), dim=-2)
            denotation = coref_gate.unsqueeze(-2).matmul(candidates).squeeze(-2)

        vis_emb = img_features_proj.view(batch_size, n_objects, d)

        return denotation, vis_emb, coref_gate


class GLTEncoder(nn.Module):
    """
    Main GLT module - runs for all lengths of spans until the representation and denotation for the root cell is computed
    """
    def __init__(self, config):
        super(GLTEncoder, self).__init__()
        self.ground = GLTTokenGrounding(config)
        self.word_energy_lin = nn.Linear(config.hidden_size, 1)

        modules = [GLTLayer(config) for _ in range(config.max_sentence_length - 1)]

        layers_to_tie = config.layers_to_tie
        if layers_to_tie:
            tie_layers(modules[0], modules[1:], config.layers_to_tie)

        self.layer = nn.ModuleList(modules)
        self.answer_nn = GLTAnswerVisualTextComp(config)

        self._contextualize_inputs_n_layers = config.contextualize_inputs_layers
        if self._contextualize_inputs_n_layers:
            self.self_att_layers = nn.ModuleList(
                [BertLayer(config) for _ in range(self._contextualize_inputs_n_layers)]
            )
        else:
            self.self_att_layers = None

    def forward(self, hidden_states, attention_mask,
                img_features: torch.Tensor = None, n_img_features: torch.Tensor = None, patches_pos=None,
                prev_sent_den=None):
        all_hidden_states = [hidden_states]
        all_attentions = (None,)
        debug_info = {}

        batch_size, seq_length, dim = hidden_states.size()
        n_objects = img_features.size(1)

        num_layers = min(len(self.layer), seq_length - 1)
        root_answers = torch.zeros(batch_size, dim).float().to(hidden_states.device)

        if self.self_att_layers is not None:
            for layer in self.self_att_layers:
                hidden_states = layer(hidden_states, attention_mask.unsqueeze(1).unsqueeze(-1))[0]

        all_denotations = None
        vis_emb = None
        obj_mask = util.get_mask_from_sequence_lengths(n_img_features, n_objects).float()

        if img_features is not None:
            denotation, vis_emb, coref_gate = self.ground(text_features=hidden_states, img_features=img_features,
                                                          patches_pos=patches_pos,
                                                          obj_mask=obj_mask,
                                                          prev_sent_den=prev_sent_den)
            all_denotations = [denotation]
            vis_root_hidden_states = denotation[:, 0]
            debug_info['coref_gate'] = coref_gate

        sentence_lengths = util.get_lengths_from_binary_sequence_mask(attention_mask)

        for i, layer_module in enumerate(self.layer[:num_layers]):
            layer_outputs = layer_module(all_hidden_states, i, all_denotations, vis_emb, obj_mask)
            hidden_states, attentions, vis_outputs, layer_debug_info = layer_outputs

            # Add last layer
            all_hidden_states += [hidden_states]
            all_attentions = all_attentions + (attentions,)

            # we should only update the root hidden states when sentence is shorter than num_layers
            # so we keep a mask of the relevant batch indices
            update_root_indices_mask = (sentence_lengths - 2 >= i).float()
            update_root_indices_mask = update_root_indices_mask.unsqueeze(1).to(hidden_states.device)

            answer, answer_debug_info = self.answer_nn(hidden_states[:, 0], vis_outputs[:, 0], vis_emb)
            layer_debug_info.update(answer_debug_info)

            root_answers = root_answers * (1 - update_root_indices_mask) + update_root_indices_mask * answer

            if img_features is not None:
                all_denotations += [vis_outputs]

                vis_root_hidden_states = vis_root_hidden_states * (1 - update_root_indices_mask) + \
                                         update_root_indices_mask * vis_outputs[:, 0]

            for key, val in layer_debug_info.items():
                debug_info[f'{key}_{i + 1}'] = val

        outputs = (root_answers, all_hidden_states, all_attentions,)
        if img_features is not None:
            outputs += (vis_root_hidden_states, vis_emb, all_denotations,)
        outputs += (debug_info,)
        return outputs  # last-layer hidden state, all hidden states, all attentions


class GLTLayer(nn.Module):
    """
    A single GLT layer
    """
    def __init__(self, config):
        super(GLTLayer, self).__init__()
        self.pair_compose = GLTPairCompose(config)

    def forward(self, all_hidden_states, current_row, all_denotations=None, vis_emb=None,
                obj_mask=None):
        # batch_size, (n-r), r, dim
        pair_compose_output = self.pair_compose(all_hidden_states, current_row, all_denotations, vis_emb,
                                                obj_mask)

        if all_denotations is not None:
            hidden_states, attentions, vis_splits, debug_info = pair_compose_output
        else:
            hidden_states, attentions, debug_info = pair_compose_output

        outputs = (hidden_states, attentions,)

        if all_denotations is not None:
            outputs += (vis_splits,)

        outputs += (debug_info,)

        return outputs


class GLTSelfOutput(nn.Module):
    """
    Same as BertSelfOutput except layer norm is optional
    """

    def __init__(self, config, dense=True, l_norm=True, dropout=True):
        super(GLTSelfOutput, self).__init__()
        self.dense = dense
        self.l_norm = l_norm
        self.dropout = dropout
        if dense:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if l_norm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if dropout:
            self.dropout = nn.Dropout(config.layer_dropout_prob)

    def forward(self, hidden_states, input_tensor=None):
        if self.dense:
            hidden_states = self.dense(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        if input_tensor is not None:
            hidden_states += input_tensor
        if self.l_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GLTPairComposeAttention(nn.Module):
    """
    Self-attend representations of two sub-spans to output a single representation ($f_h)
    """
    def __init__(self, config):
        super(GLTPairComposeAttention, self).__init__()
        self.att_score_l = nn.Linear(config.hidden_size, 1)
        self.att_score_r = nn.Linear(config.hidden_size, 1)
        self.lin_l = nn.Linear(config.hidden_size, config.hidden_size)
        self.lin_r = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hs_l = hidden_states[:, 0]
        hs_r = hidden_states[:, 1]
        weights_l = self.att_score_l(hs_l)
        weights_r = self.att_score_r(hs_r)
        self_weights = torch.softmax(torch.cat((weights_l, weights_r), dim=-1), dim=-1)
        values = torch.stack((
            self.lin_l(hs_l),
            self.lin_r(hs_r)
        ), dim=1)
        self_outputs = self_weights.unsqueeze(-2).matmul(values).squeeze(-2)

        return self_outputs, self_weights


class GLTPairCompose(nn.Module):
    """
    The module compositionally computes $h$ and $d$ for a span given all possible sub-spans splits
    """
    def __init__(self, config):
        super(GLTPairCompose, self).__init__()
        self._non_compositional_reps = config.non_compositional_reps
        if config.non_compositional_reps:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, batch_first=True,
                                bidirectional=True)

        self.attention = GLTPairComposeAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = GLTSelfOutput(config, dropout=True, dense=False, l_norm=True)

        self.weighted_output = GLTSelfOutput(config, dense=False, dropout=False)

        if config.grounded:
            self.meaning_query_2 = nn.Linear(config.hidden_size, 2)

            self.control_gate = None
            n_options = 0
            if config.control_gate_add_skip:
                n_options += 1
            if config.control_gate_add_intersect:
                n_options += 1
            if config.control_gate_add_union:
                n_options += 1
            if config.control_gate_add_vis:
                n_options += 1
                if config.control_gate_add_extra_vis_module_left_branching:
                    n_options += 1
            if n_options == 0:
                raise (AttributeError("At least one module must be added"))
            self.control_gate = nn.Linear(config.hidden_size, n_options)
            self.control_gate_add_skip = config.control_gate_add_skip
            self.control_gate_add_union = config.control_gate_add_union
            self.control_gate_add_intersect = config.control_gate_add_intersect
            self.control_gate_add_vis = config.control_gate_add_vis
            self.control_gate_add_extra_vis_module_left_branching = config.control_gate_add_extra_vis_module_left_branching
            self.control_gate_set_vis_left_branching = config.control_gate_set_vis_left_branching

            if self.control_gate_add_vis:
                self.vis_text_text_comp = GLTVisualTextComp(config)
                self.constt_rep_lin = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.layer_dropout_prob)

        self.constt_energy = self.lin = nn.Linear(config.hidden_size, 1)

    def forward(self, all_hidden_states, current_row, all_denotations=None, vis_emb=None, obj_mask=None):
        debug_info = {}

        pairs_text = self.get_splits(all_hidden_states, current_row)

        batch_size, seq_length, num_splits, _, dim = pairs_text.size()

        pairs_flat = pairs_text.view(batch_size * seq_length * num_splits, 2, dim)
        pairs_with_embeddings = pairs_flat

        # compose
        pairs_composed, compose_attentions = self.attention(pairs_with_embeddings)
        debug_info['compose_attentions'] = compose_attentions.unsqueeze(0).view(batch_size, -1, 2)

        intermediate_output = self.intermediate(pairs_composed)
        pairs_composed = self.output(intermediate_output, pairs_composed)

        layer_output_ff = pairs_composed.view(batch_size, seq_length, num_splits, dim)
        energies = self.constt_energy(layer_output_ff).squeeze(-1)

        attentions = energies.softmax(dim=-1)
        attentions_expnd_pair = attentions.unsqueeze(-2).expand(-1, -1, 2, -1)

        if self._non_compositional_reps:
            # we add 2 to current_row since in the first call to this function, current_row is 0, and length to be
            # calculated is 2
            lstm_seq_len = current_row + 2
            non_comp_sequences = [all_hidden_states[0][:, i:i + lstm_seq_len] for i in range(seq_length)]
            non_comp_sequences = torch.stack(non_comp_sequences, dim=1)
            non_comp_sequences_reps = self.lstm(non_comp_sequences.view(-1, lstm_seq_len, dim))[0]
            non_comp_sequences_reps = non_comp_sequences_reps.view(batch_size, seq_length, lstm_seq_len, 2, dim)
            # sum bi-direction reps (last of forward and first of backwards)
            constt_rep = non_comp_sequences_reps[:, :, lstm_seq_len - 1, 0] + non_comp_sequences_reps[:, :, 0, 1]
        else:
            constt_weighted = attentions.unsqueeze(-2).matmul(layer_output_ff).squeeze(-2)
            constt_rep = self.weighted_output(constt_weighted)

        outputs = (constt_rep, attentions,)

        if all_denotations is not None:
            n_objects = all_denotations[0].size(-1)
            pairs_den = self.get_splits(all_denotations, current_row)
            constt_den = attentions_expnd_pair.unsqueeze(-2).matmul(pairs_den.transpose(2, 3)).squeeze(-2)

            pair_probs = self.meaning_query_2(constt_rep).softmax(dim=-1).unsqueeze(-1)

            pair_probs_exp = pair_probs.expand(-1, -1, -1, n_objects).transpose(-2, -1)

            composed_den = [
                pair_probs_exp.unsqueeze(-2).matmul(constt_den.transpose(-2, -1).unsqueeze(-1)).squeeze(-1).squeeze(-1)]
            if self.control_gate is not None:
                intersection = torch.min(torch.stack((constt_den[:, :, 0], constt_den[:, :, 1]), dim=-1), dim=-1)[0]
                union = torch.max(torch.stack((constt_den[:, :, 0], constt_den[:, :, 1]), dim=-1), dim=-1)[0]

                if self.control_gate_add_vis:
                    constt_rep_lin = self.constt_rep_lin(constt_rep)
                    left_m = right_m = constt_rep_lin

                    m_1_den, m_2_den = self.vis_text_text_comp(left_m, right_m, vis_emb, constt_den, obj_mask,
                                                               both_directions=self.control_gate_add_extra_vis_module_left_branching,
                                                               left_branch=self.control_gate_set_vis_left_branching)

                control_candidates = []

                if self.control_gate_add_skip:
                    control_candidates += composed_den
                if self.control_gate_add_intersect:
                    control_candidates.append(intersection)
                if self.control_gate_add_union:
                    control_candidates.append(union)
                if self.control_gate_add_vis:
                    control_candidates.append(m_1_den)
                if self.control_gate_add_extra_vis_module_left_branching:
                    control_candidates.append(m_2_den)

                all_possible_dens = torch.stack(control_candidates, dim=-1)

                intersect_logits = self.control_gate(constt_rep)
                control_gate = torch.softmax(intersect_logits, dim=-1)
                control_gate_exp = control_gate.unsqueeze(-2).expand(-1, -1, n_objects, -1)
                composed_den = all_possible_dens.unsqueeze(-2).matmul(control_gate_exp.unsqueeze(-1)).squeeze(
                    -1).squeeze(-1)
                debug_info['control_gate'] = control_gate

            outputs += (composed_den,)

        outputs += (debug_info,)

        return outputs

    @staticmethod
    def get_splits(all_hidden_states, current_row):
        """
        Fetch all possible splits (CKY-based)
        """
        batch_size, seq_len, dim = all_hidden_states[0].size()

        # add 1 to row number since we started from second row
        r = current_row + 1

        pairs = []

        for col in range(seq_len - r):
            for constituent_number in range(r):
                left = all_hidden_states[constituent_number][:, col]
                right = all_hidden_states[r - 1 - constituent_number][:, col + constituent_number + 1]

                pairs.append(left)
                pairs.append(right)

        pairs = torch.stack(pairs).view((seq_len - r, r, 2, batch_size, dim)).permute(3, 0, 1, 2, 4).contiguous()

        return pairs


class GLTVisualTextComp(nn.Module):
    """
    The VISUAL module
    """
    def __init__(self, config):
        super(GLTVisualTextComp, self).__init__()

        self.comp_func = GLTVisualTextCompFunc(config)

    def forward(self, left_m, right_m, vis_emb, pairs_den, obj_mask, both_directions: bool, left_branch: bool):
        assert not (left_branch and both_directions)
        batch_size, seq_length, _, n_objects = pairs_den.size()
        vis_emb_exp = vis_emb.unsqueeze(1).expand(-1, seq_length, -1, -1)
        obj_mask = obj_mask.unsqueeze(1)

        left_den, right_den = pairs_den[:, :, 0], pairs_den[:, :, 1]
        m_1_den = None
        m_2_den = None

        if not left_branch:
            m_1_den = self.comp_func(vis_emb_exp,
                                     left_m,
                                     left_den,
                                     right_den,
                                     obj_mask)

        if both_directions or left_branch:
            m_2_den = self.comp_func(vis_emb_exp,
                                       right_m,
                                       right_den,
                                       left_den,
                                       obj_mask)
            if left_branch:
                m_1_den = m_2_den
                m_2_den = None

        return m_1_den, m_2_den


class GLTVisualTextCompFunc(nn.Module):
    """
    The actual function used by the VISUAL module
    """
    def __init__(self, config):
        super(GLTVisualTextCompFunc, self).__init__()
        self.vis_text_ff1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.visual_module_dropout_prob)

        self.same_obj_emb = nn.Embedding(2, config.hidden_size)

        self.den_emb = nn.Embedding(2, config.hidden_size)

        self.vis_text_ff2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.vis_text_ff3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.vis_text_ff4 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, vis_emb, meaning_query, left_den, right_den, obj_mask):
        right_vis_emb = right_den.unsqueeze(-2).matmul(vis_emb)
        if len(meaning_query.size()) != len(vis_emb.size()):
            meaning_query = meaning_query.unsqueeze(-2)

        dims = len(meaning_query.size())
        expand_by = (-1,) * (dims - 1) + (meaning_query.size(-1),)
        # an embedding for each object, indicating if it has the same index as the "right hand side" we have from den
        same_obj_emb = self.same_obj_emb.weight[0] * right_den.unsqueeze(-1).expand(*expand_by) + \
                       self.same_obj_emb.weight[1] * (1 - right_den).unsqueeze(-1).expand(*expand_by)

        # an embedding for each object, indicating its denotation on the "left hand side"
        left_den_emb = self.den_emb.weight[0] * left_den.unsqueeze(-1).expand(*expand_by) + \
                       self.den_emb.weight[1] * (1 - left_den).unsqueeze(-1).expand(*expand_by)

        meaning_query_exp = meaning_query.expand_as(vis_emb)
        left_input = vis_emb + meaning_query_exp + same_obj_emb
        left_input += left_den_emb
        left_m = self.vis_text_ff1(left_input)
        left_m = self.vis_text_ff3(self.dropout(self.activation(left_m)))
        right_m = self.vis_text_ff2(right_vis_emb + meaning_query)
        right_m = self.vis_text_ff4(self.dropout(self.activation(right_m)))

        right_m = right_m.expand_as(left_m)
        return torch.sigmoid(left_m.unsqueeze(-2).matmul(right_m.unsqueeze(-1)).squeeze(-1).squeeze(-1)) * obj_mask


class GLTAnswerVisualTextComp(nn.Module):
    """
    This module takes the textual and visual representation of the root node, and returns a final representation, ready
    to be classified
    """
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.answer_comp_dropout_prob)

        if config.grounded:
            # classifier input is both text and visual representation
            input_dim = config.hidden_size * 2
        else:
            input_dim = config.hidden_size

        if config.answer_pooler:
            self.pooler = GLTPooler(config, input_dim)
        else:
            self.pooler = lambda x: x

    def forward(self, meaning_query, vis_outputs=None, vis_emb=None):
        debug_info = {}

        if vis_outputs is not None:
            weighted_vis_emb = vis_outputs.unsqueeze(-2).matmul(vis_emb).squeeze(-2)

            classifier_input = torch.cat((
                weighted_vis_emb,
                meaning_query
            ), dim=-1)
        else:
            classifier_input = meaning_query

        hidden_rep = self.dropout(self.pooler(classifier_input))

        return hidden_rep, debug_info
