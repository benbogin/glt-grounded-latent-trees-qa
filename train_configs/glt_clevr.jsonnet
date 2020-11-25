local random_seed = 42;
local max_sentence_length=30;

{
  "random_seed": random_seed,
  "numpy_seed": random_seed,
  "pytorch_seed": random_seed,
  "dataset_reader": {
    "type": "clevr",
    "limit": -1,
    "data_dir": "data/",
    "remove_stop_words": true,
    "split_coreferences": true
  },
  "train_data_path": "data/train.jsonl",
  "validation_data_path": "data/val.jsonl",
  "model": {
    "type": "glt",
    // see `models/glt.py` for descriptions of all parameters
    "max_sentence_length": max_sentence_length,
    "hidden_size": 400,
    "input_img_dim": 2048,
    "num_attention_heads": 1,
    "use_position_embeddings": false,
    "classifier_dropout_prob": 0.0,
    "layer_dropout_prob": 0.25,
    "answer_comp_dropout_prob": 0.25,
    "transformer_dropout_prob": 0.0,
    "visual_module_dropout_prob": 0.0,
    "visual_self_attention_layers": 1,
    "glove_path": null,
    "non_compositional_reps": false,
    "contextualize_inputs_layers": 0,
    "control_gate_add_union": true,
    "control_gate_add_intersect": true,
    "control_gate_add_skip": true,
    "control_gate_add_vis": true,
    "control_gate_set_vis_left_branching": false,
    "control_gate_add_extra_vis_module_left_branching": false,
    "answer_pooler": true,
    "layers_to_tie": [
        "pair_compose.intermediate.dense",
        "pair_compose.control_gate",
        "pair_compose.attention",
        "pair_compose.constt_energy",
        "pair_compose.constt_rep_lin",
        "pair_compose.vis_text_text_comp.comp_func",
        "pair_compose.meaning_query_2",
    ]
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 100,
    "sorting_keys": [['tokens', 'tokens___tokens']],
    "biggest_batch_first": false
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size": 120
  },
  "trainer": {
    "type": "callback",
    "num_epochs": 40,
    "cuda_device": 0,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
      "weight_decay": 0.075,
      "parameter_groups": [
        [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
        [["pair_compose.attention.*.weight", "pair_compose.intermediate.dense.weight","encoder.answer_nn.*.weight"], {"weight_decay": 0.15}]
      ]
    },
    "callbacks": [
        "log_to_tensorboard",
        "validate",
        "alternative_metric",
        {
            "type": "checkpoint",
            "checkpointer": {
                "num_serialized_models_to_keep": 1
            }
        },
        {
            "type": "track_metrics",
            "validation_metric": "+accuracy"
        },
        {
            "type": "clip_grad_norm",
            "max_grad_norm": 0.6
        },
    ]
  }
}