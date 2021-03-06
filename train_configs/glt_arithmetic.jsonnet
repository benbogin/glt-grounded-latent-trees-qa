local epoch_size = 10000;
local validation_epoch_size = 250;
local random_seed = 42;

{
  "random_seed": random_seed,
  "numpy_seed": random_seed,
  "pytorch_seed": random_seed,
  "dataset_reader": {
    "type": "arithmetic",
    "epoch_size": epoch_size,
    "validation_epoch_size": validation_epoch_size,
    "random_seed": random_seed,
    "train_max_num_numbers": 8,
    "len_split_num_numbers": 10,
    "max_answer": 100,
    "randomly_pad": false,
    "randomly_pad_end": false,
    "split_by_operation_pos": true,
    "split_by_operation_order_num_keys": 3,
    "math_operations": ["+", "*"]
  },
  "train_data_path": "train",
  "validation_data_path": "val",
  "vocabulary": {
    "tokens_to_add": {
        "labels": [std.toString(a) for a in std.range(0, 100)]
    }
  },
  "model": {
    "type": "glt_ungrounded",
    "max_sentence_length": 19,
    "hidden_size": 300,
    "use_position_embeddings": false,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_dropout_prob": 0.25,
    "answer_comp_dropout_prob": 0.0,
    "tie_layer_norm": true,
    "answer_pooler": false,
    "layers_to_tie": [
        "pair_compose.attention",
        "pair_compose.intermediate.dense",
        "pair_compose.constt_energy",
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 100,
    "instances_per_epoch": epoch_size
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size": 100,
    "instances_per_epoch": validation_epoch_size * 3  // both iid and 2 ood
  },
  "trainer": {
    "type": "callback",
    "num_epochs": 300,
    "cuda_device": 0,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-3,
      "weight_decay": 0.1,
      "parameter_groups": [
        [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}]
      ]
    },
    "callbacks": [
        "log_to_tensorboard",
        "validate",
        {
            "type": "checkpoint",
            "checkpointer": {
                "num_serialized_models_to_keep": 2
            },
        },
        {
            "type": "track_metrics",
            "validation_metric": "+accuracy"
        },
        {
            "type": "clip_grad_norm",
            "max_grad_norm": 1.0
        },
    ]
  }
}