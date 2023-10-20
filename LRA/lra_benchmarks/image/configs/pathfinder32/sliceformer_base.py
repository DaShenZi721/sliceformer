# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.image.configs.pathfinder32 import base_pathfinder32_config

NUM_EPOCHS = 200
TRAIN_EXAMPLES = 160000
VALID_EXAMPLES = 20000

def get_config():
  """Get the hyperparameter configuration."""
  config = base_pathfinder32_config.get_config()
  config.model_type = "sliceformer"

  config.batch_size = 640

  config.save_checkpoints = False
  config.restore_checkpoints = False
  config.checkpoint_freq = (TRAIN_EXAMPLES // config.batch_size)

  config.random_seed = 3

  config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
  config.num_train_steps = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS
  config.num_eval_steps = VALID_EXAMPLES // config.batch_size

  config.weight_decay = 0.0
  config.learning_rate = 0.0005
  config.model.dropout_rate = 0.1
  config.model.attention_dropout_rate = 0.1
  config.grad_clip_norm = None

  config.model.num_layers = 4
  config.model.num_heads = 8
  config.model.emb_dim = 320
  config.model.qkv_dim = 320
  config.model.mlp_dim = 320

  config.model.classifier_pool = 'CLS'
  config.model.learn_pos_emb = True
  config.trial = 0  # dummy for repeated runs.
  return config


def get_hyper(hyper):
  return hyper.product([])
