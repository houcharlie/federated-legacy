import collections
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os
from PIL import Image

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=True)
emnist_train = emnist_train.create_tf_dataset_from_all_clients()
#print(len(list(emnist_train)))
dataset_iterator = iter(emnist_train)
if not os.path.exists('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/train/'):
  os.makedirs('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/train/')
  os.makedirs('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/test/')
for i, sample in enumerate(dataset_iterator):
  image = sample['pixels'].numpy()
  im = Image.fromarray(image)
  im = im.convert("L")
  label = int(sample['label'].numpy())
  if not os.path.exists('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/train/{0}'.format(label)):
    os.makedirs('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/train/{0}'.format(label))
  im.save('/pylon5/ir5fpvp/houc/datasets-image/EMNIST/train/{0}/{1}.png'.format(label, i))
  if i % 1000 == 0:
    print(i, flush = True)
