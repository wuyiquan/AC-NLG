# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def query_aware_attention(rqs_states, art_states, rqs_padding_mask):
  """
  Args:
    rqs_states: 3D Tensor [batch_size, <=max_enc_steps, 2*hidden_dim].
    art_states: 3D Tensor [batch_size, <=max_enc_steps, 2*hidden_dim].
    rqs_padding_mask: 2D Tensor [batch_size x max_enc_steps] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
    cell: rnn_cell.RNNCell defining the cell function and size.
    art_lens: [batch_size]
    cell: rnn_cell.RNNCell defining the cell function and size.

  Returns:
    outputs: 3D Tensor [batch_size, <=max_enc_steps, hidden_dim]. The output vectors.
  """
  with variable_scope.variable_scope("query_aware_attention") as scope:
    batch_size = rqs_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = rqs_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined

    # Reshape encoder_states (need to insert a dim)
    rqs_states = tf.expand_dims(rqs_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)

    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    encoder_features = nn_ops.conv2d(rqs_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)

    # Get the weight vectors v
    v = variable_scope.get_variable("v", [attention_vec_size])

    def attention(art_state):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        art_state: state of the art

      Returns:
        context_vector: weighted sum of rqs_states
      """
      with variable_scope.variable_scope("Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(art_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= rqs_padding_mask # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize


        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3]) # calculate e

        # Calculate attention distribution
        attn_dist = masked_attention(e)

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * rqs_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector

    outputs = []
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.

    for i, inp in enumerate(art_states):
      # tf.logging.info("Adding query_aware_attention timestep %i of %i", i, len(art_states))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      # Run the attention mechanism.
      context_vector = attention(inp)

      x = linear([inp] + [context_vector], input_size, True)
      outputs.append(x)

    outputs = tf.stack(outputs, axis=1)


    return outputs



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
