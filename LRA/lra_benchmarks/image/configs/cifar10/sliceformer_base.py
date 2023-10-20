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

from lra_benchmarks.image.configs.cifar10 import base_cifar10_config

NUM_EPOCHS = 200
TRAIN_EXAMPLES = 45000
VALID_EXAMPLES = 10000

def get_config():
  """Get the hyperparameter configuration."""
  config = base_cifar10_config.get_config()
  config.model_type = "sliceformer"

  config.batch_size = 192

  config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
  config.num_train_steps = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS
  config.num_eval_steps = VALID_EXAMPLES // config.batch_size
  config.grad_clip_norm = None

  config.save_checkpoints = False
  config.restore_checkpoints = False

  config.learning_rate = .005

  config.model.num_layers = 6
  config.model.classifier_pool = "CLS"
  config.model.emb_dim = 96
  config.model.mlp_dim = 128
  config.model.num_heads = 8
  config.model.qkv_dim = 48

  config.weight_decay = 0.0
  config.model.dropout_rate = 0.2
  config.model.attention_dropout_rate = 0.1
  return config


def get_hyper(hyper):
  return hyper.product([])
