# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to replicate experiments in https://arxiv.org/abs/1911.06679."""

from absl import app
from absl import flags
# pylint: disable=no-name-in-module
# pylint: disable=import-error
from tensorflow_federated.python.research.utils import utils_impl 

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  target = '//tensorflow_federated/python/research/gans/experiments/emnist:train'
  executable = 'cd /home/houc/federated/tensorflow_federated/python/research/gans; bazel --output_user_root=/pylon5/ir5fpvp/houc/tmp/bazel run {} --'.format(target)
  '''
  grid_iter = utils_impl.iter_grid({
      'filtering': ['by_user'],
      'invert_imagery_probability': ['0p0', '0p5'],
      'accuracy_threshold': ['lt0p882', 'gt0p939'],
      'num_client_disc_train_steps': [6],
      'num_server_gen_train_steps': [6],
      'num_clients_per_round': [10],
      'num_rounds': [1000],
      'use_dp': [True],
      'dp_l2_norm_clip': [0.1],
      'dp_noise_multiplier': [0.01],
      'num_rounds_per_eval': [10],
      'num_rounds_per_save_images': [10]
  })
  '''
  '''
  grid_iter = utils_impl.iter_grid({
      'filtering': ['None'],
      'invert_imagery_probability': ['0p0'],
      'num_client_disc_train_steps': [6],
      'num_server_gen_train_steps': [6],
      'num_clients_per_round': [10],
      'num_rounds': [1000],
      'num_rounds_per_eval': [10],
      'num_rounds_per_save_images': [10],
      'generator_status': ['CEN', 'FED'],
      'discriminator_status': ['FED']
  })
  '''
  grid_iter = utils_impl.iter_grid({
      'filtering': ['None'],
      'invert_imagery_probability': ['0p0'],
      'num_client_disc_train_steps': [60],
      'num_server_gen_train_steps': [60],
      'num_clients_per_round': [1],
      'num_rounds': [10000],
      'num_rounds_per_eval': [10],
      'num_rounds_per_save_images': [10],
      'status':['CEN_CEN']
  })
  utils_impl.launch_experiment(
      executable,
      grid_iter,
      root_output_dir='/home/houc/fedgans_res_model_v1',
      short_names={
          'filtering': 'filt',
          'invert_imagery_probability': 'inv_lik',
          'accuracy_threshold': 'acc',
          'num_client_disc_train_steps': 'n_disc',
          'num_server_gen_train_steps': 'n_gen',
          'dp_l2_norm_clip': 'dp_clip',
          'dp_noise_multiplier': 'dp_noise',
          'num_rounds_per_eval': 'n_rds_eval',
          'num_rounds_per_save_images': 'n_rds_images',
      },
      max_workers=1)
  print('Experiments launched.')


if __name__ == '__main__':
  app.run(main)
