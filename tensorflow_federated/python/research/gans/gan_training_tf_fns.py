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
"""TensorFlow training code for Federated GANs.

This code is intended to only use vanilla TensorFlow (no TFF dependency); it is
wired together into a federated computation in gan_training_tff_fns.py. The one
exception is some handling for conversion from Struct, which should go
away when b/130724878 is fixed.
"""

import attr
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.research.gans import gan_losses
from tensorflow_federated.python.tensorflow_libs import tensor_utils


def assert_no_anon_tuples(x):
  """Checks that a nested structure has no Structs at the leaves."""

  def check_anon(t):
    if 'Struct' in str(type(t)):
      raise ValueError('Found Struct:\n', t)
    return None

  tf.nest.map_structure(check_anon, x)
  return x


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class FromServer(object):
  """Container for data that is broadcast from the server to clients.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()
  state_gen_optimizer_weights = attr.ib()
  state_disc_optimizer_weights = attr.ib()
  counters = attr.ib()



# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=False)
class ServerState(object):
  """Container for all server state that must pass from round to round.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
    counters: Aggregated training counters.
    dp_averaging_state: State (possibly empty) of the dp_averaging_fn.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()
  state_gen_optimizer_weights = attr.ib()
  state_disc_optimizer_weights = attr.ib()
  generator_diff = attr.ib()
  discriminator_diff = attr.ib()
  counters = attr.ib()
  dp_averaging_state = attr.ib(default=None)


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Container for data that is sent from clients back to the server..

  Attributes:
    discriminator_weights_delta: Update for the discriminator model, in the
      order of tf.keras.Model.weights`.
    update_weight: Weight to be associated with the update.
    counters: Metrics that are summed across clients.
  """
  counters = attr.ib()
  discriminator_weights_delta = attr.ib()
  generator_weights_delta = attr.ib()
  state_gen_opt_delta = attr.ib()
  state_disc_opt_delta = attr.ib()
  update_weight = attr.ib()
  update_weight_D = attr.ib()
  update_weight_G = attr.ib()  


def _weights(model):
  """Returns tensors of model weights, in the order of the variables."""
  return tff.learning.ModelWeights.from_model(model)

def initialize_optimizer_vars(model: tf.keras.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, model.trainable_variables)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model.trainable_weights)
  optimizer.apply_gradients(grads_and_vars, name='server_update')

  assert optimizer.variables()

@tf.function
def client_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    gen_inputs_ds: tf.data.Dataset,
    real_data_ds: tf.data.Dataset,
    from_server: FromServer,
    # Python arguments bound to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    state_gen_optimizer: tf.keras.optimizers.Optimizer,
    state_disc_optimizer: tf.keras.optimizers.Optimizer) -> ClientOutput:
  """The computation to run on the client, training the discriminator.

  Args:
    gen_inputs_ds: A `tf.data.Dataset` of generator_inputs.
    real_data_ds: A `tf.data.Dataset` of data from the real distribution.
    from_server: A `FromServer` object, including the current model weights.
    generator:  The generator.
    discriminator: The discriminator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.

  Returns:
    A `ClientOutput` object.
  """
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(generator),
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(discriminator),
                        from_server.discriminator_weights)
  tf.nest.map_structure(conditioned_assign, state_disc_optimizer.variables(), from_server.state_disc_optimizer_weights)
  tf.nest.map_structure(conditioned_assign, state_gen_optimizer.variables(), from_server.state_gen_optimizer_weights)
  num_examples = tf.constant(0)
  loss_fns = gan_losses.WassersteinGanLossFns()
  gen_inputs_and_real_data = tf.data.Dataset.zip((gen_inputs_ds, real_data_ds))
  for gen_inputs, real_data in gen_inputs_and_real_data:
    # It's possible that real_data and gen_inputs have different batch sizes.
    # For calculating the discriminator loss, it's desirable to have equal-sized
    # contributions from both the real and fake data. Also, it's necessary if
    # using the Wasserstein gradient penalty (where a difference is taken b/w
    # the real and fake data). So here we reduce to the min batch size. This
    # also ensures num_examples properly reflects the amount of data trained on.
    min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(gen_inputs)[0])
    real_data = real_data[0:min_batch_size]
    gen_inputs = gen_inputs[0:min_batch_size]
    with tf.GradientTape() as tape:
      loss = loss_fns.discriminator_loss(generator, discriminator, gen_inputs, real_data)
    grads = tape.gradient(loss, discriminator.trainable_variables)
    grads_and_vars = zip(grads, discriminator.trainable_variables)
    state_disc_optimizer.apply_gradients(grads_and_vars)
    num_examples += min_batch_size

  state_disc_opt_delta = tf.nest.map_structure(tf.subtract, 
                                        tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_disc_optimizer.variables()),
                                        from_server.state_disc_optimizer_weights)
  # should be zero
  state_gen_opt_delta = tf.nest.map_structure(tf.subtract, 
                                        tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_gen_optimizer.variables()),
                                        from_server.state_gen_optimizer_weights)
  weights_delta = tf.nest.map_structure(tf.subtract, discriminator.trainable_weights,
                                        from_server.discriminator_weights.trainable)
  weights_delta, has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  update_weight = tf.cast(num_examples, tf.float32)
  # Zero out the weight if there are any non-finite values.
  # TODO(b/122071074): federated_mean might not do the right thing if
  # all clients have zero weight.
  update_weight = tf.cond(
      tf.equal(has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))
  return ClientOutput(
      discriminator_weights_delta=weights_delta,
      generator_weights_delta=weights_delta,
      state_disc_opt_delta=state_disc_opt_delta,
      state_gen_opt_delta=state_gen_opt_delta,
      update_weight_D=update_weight,
      update_weight_G=update_weight,
      update_weight=update_weight,
      counters={'num_discriminator_train_examples': num_examples})
@tf.function
def client_computation_fedadam(
    # Tensor/Dataset arguments that will be supplied by TFF:
    gen_inputs_ds: tf.data.Dataset,
    real_data_ds: tf.data.Dataset,
    from_server: FromServer,
    # Python arguments bound to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    state_gen_optimizer: tf.keras.optimizers.Optimizer,
    state_disc_optimizer: tf.keras.optimizers.Optimizer) -> ClientOutput:
  """The computation to run on the client, training the discriminator.

  Args:
    gen_inputs_ds: A `tf.data.Dataset` of generator_inputs.
    real_data_ds: A `tf.data.Dataset` of data from the real distribution.
    from_server: A `FromServer` object, including the current model weights.
    generator:  The generator.
    discriminator: The discriminator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.

  Returns:
    A `ClientOutput` object.
  """
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(generator),
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(discriminator),
                        from_server.discriminator_weights)
  tf.nest.map_structure(conditioned_assign, state_disc_optimizer.variables(), from_server.state_disc_optimizer_weights)
  tf.nest.map_structure(conditioned_assign, state_gen_optimizer.variables(), from_server.state_gen_optimizer_weights)
  num_examples = tf.constant(0)
  loss_fns = gan_losses.WassersteinGanLossFns()
  gen_inputs_and_real_data = tf.data.Dataset.zip((gen_inputs_ds, real_data_ds))

  sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=1)
  for gen_inputs, real_data in gen_inputs_and_real_data:
    # It's possible that real_data and gen_inputs have different batch sizes.
    # For calculating the discriminator loss, it's desirable to have equal-sized
    # contributions from both the real and fake data. Also, it's necessary if
    # using the Wasserstein gradient penalty (where a difference is taken b/w
    # the real and fake data). So here we reduce to the min batch size. This
    # also ensures num_examples properly reflects the amount of data trained on.
    min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(gen_inputs)[0])
    real_data = real_data[0:min_batch_size]
    gen_inputs = gen_inputs[0:min_batch_size]
    with tf.GradientTape() as tape:
      loss = loss_fns.discriminator_loss(generator, discriminator, gen_inputs, real_data)
    grads = tape.gradient(loss, discriminator.trainable_variables)
    grads_and_vars = zip(grads, discriminator.trainable_variables)
    sgd_optimizer.apply_gradients(grads_and_vars)
    num_examples += min_batch_size

  state_disc_opt_delta = tf.nest.map_structure(tf.subtract, 
                                        tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_disc_optimizer.variables()),
                                        from_server.state_disc_optimizer_weights)
  # should be zero
  state_gen_opt_delta = tf.nest.map_structure(tf.subtract, 
                                        tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_gen_optimizer.variables()),
                                        from_server.state_gen_optimizer_weights)
  weights_delta = tf.nest.map_structure(tf.subtract, discriminator.trainable_weights,
                                        from_server.discriminator_weights.trainable)
  weights_delta, has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  update_weight = tf.cast(num_examples, tf.float32)
  # Zero out the weight if there are any non-finite values.
  # TODO(b/122071074): federated_mean might not do the right thing if
  # all clients have zero weight.
  update_weight = tf.cond(
      tf.equal(has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))
  return ClientOutput(
      discriminator_weights_delta=weights_delta,
      generator_weights_delta=weights_delta,
      state_disc_opt_delta=state_disc_opt_delta,
      state_gen_opt_delta=state_gen_opt_delta,
      update_weight_D=update_weight,
      update_weight_G=update_weight,
      update_weight=update_weight,
      counters={'num_discriminator_train_examples': num_examples})


def server_initial_state(generator, discriminator, gen_opt, disc_opt, dp_averaging_state=None):
  """Returns the initial state of the server."""
  return ServerState(
      generator_weights=_weights(generator),
      discriminator_weights=_weights(discriminator),
      generator_diff = tf.nest.map_structure(tf.zeros_like, generator.trainable_weights),
      discriminator_diff = tf.nest.map_structure(tf.zeros_like, discriminator.trainable_weights),
      state_gen_optimizer_weights=tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), gen_opt.variables()),
      state_disc_optimizer_weights=tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), disc_opt.variables()),
      counters={
          'num_discriminator_train_examples': tf.constant(0),
          'num_generator_train_examples': tf.constant(0),
          'num_rounds': tf.constant(0),
      },
      dp_averaging_state=dp_averaging_state)

@tf.function
def conditioned_assign(a,b):
  if a.dtype == tf.int64:
    a.assign(tf.cast(b, tf.int64))
  else:
    a.assign(b)
@tf.function
def server_computation_fedadam(
    # Tensor/Dataset arguments that will be supplied by TFF:
    server_state: ServerState,
    gen_inputs_ds: tf.data.Dataset,
    client_output: ClientOutput,
    # Python arguments to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    state_gen_optimizer: tf.keras.optimizers.Optimizer,
    state_disc_optimizer: tf.keras.optimizers.Optimizer,
    # Not an argument bound at TFF computation construction time, but placed
    # last so that it can be defaulted to empty tuple (for non-DP use cases).
    new_dp_averaging_state=()
) -> ServerState:
  """The computation to run on the server, training the generator.

  Args:
    server_state: The initial `ServerState` for the round.
    gen_inputs_ds: An infinite `tf.data.Dataset` of inputs to the `generator`.
    client_output: The (possibly aggregated) `ClientOutput`.
    generator:  The generator.
    discriminator: The discriminator.
    server_disc_update_optimizer: Optimizer used to `apply_gradients` based on
      the client_output delta.
    train_generator_fn: A function which takes the two networks and generator
      input and trains the generator.
    new_dp_averaging_state: The updated state of the DP averaging aggregator.

  Returns:
    An updated `ServerState` object.
  """
  # A tf.function can't modify the structure of its input arguments,
  # so we make a semi-shallow copy:
  server_state = attr.evolve(server_state, counters=dict(server_state.counters))
  
  tf.nest.map_structure(conditioned_assign, state_gen_optimizer.variables(), server_state.state_gen_optimizer_weights)
  tf.nest.map_structure(conditioned_assign, state_disc_optimizer.variables(), server_state.state_disc_optimizer_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(generator),
                        server_state.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(discriminator),
                        server_state.discriminator_weights)
  server_gen_update_optimizer = tf.keras.optimizers.SGD(learning_rate=1)
  server_disc_update_optimizer = tf.keras.optimizers.SGD(learning_rate=1)


  delta = client_output.discriminator_weights_delta
  tf.nest.assert_same_structure(delta, discriminator.trainable_weights)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), delta,
                                         discriminator.trainable_weights)
  state_disc_optimizer.apply_gradients(
      grads_and_vars, name='server_update_disc')

  for k, v in client_output.counters.items():
    server_state.counters[k] += v

  # Update the state of the DP averaging aggregator.
  server_state.dp_averaging_state = new_dp_averaging_state

  gen_examples_this_round = tf.constant(0)

  loss_fns = gan_losses.WassersteinGanLossFns()
  for gen_inputs in gen_inputs_ds:  # Compiled by autograph.
    with tf.GradientTape() as tape2:
      loss2 = loss_fns.generator_loss(generator, discriminator, gen_inputs)
    grads2 = tape2.gradient(loss2, generator.trainable_variables)
    grads_and_vars2 = zip(grads2, generator.trainable_variables)
    state_gen_optimizer.apply_gradients(grads_and_vars2)
    gen_examples_this_round += tf.shape(gen_inputs)[0]

  

  G_change = tf.nest.map_structure(tf.subtract, generator.trainable_weights,
                                        server_state.generator_weights.trainable)
  D_change = tf.nest.map_structure(tf.subtract, discriminator.trainable_weights,
                                        server_state.discriminator_weights.trainable)                                        
  server_state.state_gen_optimizer_weights = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_gen_optimizer.variables())
  server_state.state_disc_optimizer_weights = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_disc_optimizer.variables())
  server_state.counters[
      'num_generator_train_examples'] += gen_examples_this_round
  server_state.generator_weights = _weights(generator)
  server_state.discriminator_weights = _weights(discriminator)
  server_state.counters['num_rounds'] += 1
  server_state.generator_diff = G_change
  server_state.discriminator_diff = D_change
  return server_state

@tf.function
def server_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    server_state: ServerState,
    gen_inputs_ds: tf.data.Dataset,
    client_output: ClientOutput,
    # Python arguments to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    state_gen_optimizer: tf.keras.optimizers.Optimizer,
    state_disc_optimizer: tf.keras.optimizers.Optimizer,
    # Not an argument bound at TFF computation construction time, but placed
    # last so that it can be defaulted to empty tuple (for non-DP use cases).
    new_dp_averaging_state=()
) -> ServerState:
  """The computation to run on the server, training the generator.

  Args:
    server_state: The initial `ServerState` for the round.
    gen_inputs_ds: An infinite `tf.data.Dataset` of inputs to the `generator`.
    client_output: The (possibly aggregated) `ClientOutput`.
    generator:  The generator.
    discriminator: The discriminator.
    server_disc_update_optimizer: Optimizer used to `apply_gradients` based on
      the client_output delta.
    train_generator_fn: A function which takes the two networks and generator
      input and trains the generator.
    new_dp_averaging_state: The updated state of the DP averaging aggregator.

  Returns:
    An updated `ServerState` object.
  """
  # A tf.function can't modify the structure of its input arguments,
  # so we make a semi-shallow copy:
  server_state = attr.evolve(server_state, counters=dict(server_state.counters))
  
  tf.nest.map_structure(conditioned_assign, state_gen_optimizer.variables(), server_state.state_gen_optimizer_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(generator),
                        server_state.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), _weights(discriminator),
                        server_state.discriminator_weights)
  server_gen_update_optimizer = tf.keras.optimizers.SGD(learning_rate=1)
  server_disc_update_optimizer = tf.keras.optimizers.SGD(learning_rate=1)


  delta = client_output.discriminator_weights_delta
  tf.nest.assert_same_structure(delta, discriminator.trainable_weights)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), delta,
                                         discriminator.trainable_weights)
  server_disc_update_optimizer.apply_gradients(
      grads_and_vars, name='server_update_disc')

  for k, v in client_output.counters.items():
    server_state.counters[k] += v

  # Update the state of the DP averaging aggregator.
  server_state.dp_averaging_state = new_dp_averaging_state

  gen_examples_this_round = tf.constant(0)

  loss_fns = gan_losses.WassersteinGanLossFns()
  for gen_inputs in gen_inputs_ds:  # Compiled by autograph.
    with tf.GradientTape() as tape2:
      loss2 = loss_fns.generator_loss(generator, discriminator, gen_inputs)
    grads2 = tape2.gradient(loss2, generator.trainable_variables)
    grads_and_vars2 = zip(grads2, generator.trainable_variables)
    state_gen_optimizer.apply_gradients(grads_and_vars2)
    gen_examples_this_round += tf.shape(gen_inputs)[0]
  # update discriminator optimizer
  delta_opt_D = client_output.state_disc_opt_delta
  updated_opt_D = tf.nest.map_structure(lambda a,b: a + b, server_state.state_disc_optimizer_weights, delta_opt_D)
  

  G_change = tf.nest.map_structure(tf.subtract, generator.trainable_weights,
                                        server_state.generator_weights.trainable)
  D_change = tf.nest.map_structure(tf.subtract, discriminator.trainable_weights,
                                        server_state.discriminator_weights.trainable)                                        
  server_state.state_gen_optimizer_weights = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), state_gen_optimizer.variables())
  server_state.state_disc_optimizer_weights = updated_opt_D
  server_state.counters[
      'num_generator_train_examples'] += gen_examples_this_round
  server_state.generator_weights = _weights(generator)
  server_state.discriminator_weights = _weights(discriminator)
  server_state.counters['num_rounds'] += 1
  server_state.generator_diff = G_change
  server_state.discriminator_diff = D_change
  return server_state

def create_train_generator_fn(gan_loss_fns: gan_losses.AbstractGanLossFns,
                              gen_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains generator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    gen_optimizer: Optimizer for training the generator.

  Returns:
    Function that executes one step of generator training.
  """
  # We check that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if gen_optimizer.variables():
    raise ValueError(
        'Expected gen_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_generator_fn(generator: tf.keras.Model,
                         discriminator: tf.keras.Model, generator_inputs):
    """Trains the generator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.

    Returns:
      The number of examples trained on.
    """

    def gen_loss():
      """Does the forward pass and computes losses for the generator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.generator_loss(generator, discriminator,
                                         generator_inputs)

    gen_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
    return tf.shape(generator_inputs)[0]

  return train_generator_fn


def create_train_discriminator_fn(
    gan_loss_fns: gan_losses.AbstractGanLossFns,
    disc_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains discriminator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    disc_optimizer: Optimizer for training the discriminator.

  Returns:
    Function that executes one step of discriminator training.
  """
  # We assert that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if disc_optimizer.variables():
    raise ValueError(
        'Expected disc_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_discriminator_fn(generator: tf.keras.Model,
                             discriminator: tf.keras.Model, generator_inputs,
                             real_data):
    """Trains the discriminator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.
      real_data: A batch of real data for the discriminator.

    Returns:
      The size of the batch.
    """

    def disc_loss():
      """Does the forward pass and computes losses for the discriminator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.discriminator_loss(generator, discriminator,
                                             generator_inputs, real_data)

    disc_optimizer.minimize(
        disc_loss, var_list=discriminator.trainable_variables)
    return tf.shape(real_data)[0]

  return train_discriminator_fn
