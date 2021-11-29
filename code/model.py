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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.keras.python.keras.utils import to_categorical

from attention_decoder import attention_decoder
from query_aware_attention import query_aware_attention
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab, pre_trained_embedding=[]):
    self._hps = hps
    self._vocab = vocab
    self._pre_trained_embedding= pre_trained_embedding

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps
    # encoder part
    self._rst_batch = tf.placeholder(tf.float32, [hps.batch_size, 1], name='rst_batch')

    self._rqs_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name='rqs_batch')
    self._rqs_lens = tf.placeholder(tf.int32, [hps.batch_size], name='rqs_lens')
    self._rqs_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name='rqs_padding_mask')

    self._art_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name='art_batch')
    self._art_lens = tf.placeholder(tf.int32, [hps.batch_size], name='art_lens')
    self._art_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name='art_padding_mask')
    # self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name='enc_padding_mask')

    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._rst_batch] = batch.rst_batch
    feed_dict[self._rqs_batch] = batch.rqs_batch
    feed_dict[self._rqs_lens] = batch.rqs_lens
    feed_dict[self._rqs_padding_mask] = batch.rqs_padding_mask

    feed_dict[self._art_batch] = batch.art_batch
    feed_dict[self._art_lens] = batch.art_lens
    feed_dict[self._art_padding_mask] = batch.art_padding_mask
    # feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.art_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len, name=''):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'+name):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st

  def _add_predictor(self, encoder_outputs):
    """Add a predictor to the graph.

    Args:
      encoder_outputs: 2D Tensor [batch_size x cell.state_size].

    Returns:
      predict_results: A tensor of shape [batch_size, 1]
    """
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    hidden_dim = 128
    num_classes = 2

    with tf.variable_scope('predictor'):
      # conv = tf.layers.conv1d(encoder_outputs, num_filters, kernel_size)
      # gmp = tf.reduce_mean(conv, reduction_indices=[1])
      fc = tf.layers.dense(encoder_outputs, hidden_dim)
      # fc = tf.contrib.layers.dropout(fc, self._keep_prob)
      fc = tf.nn.relu(fc)
      logits = tf.layers.dense(fc, num_classes)
      self.p_j = tf.nn.softmax(logits)
      y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别
    return logits, y_pred_cls

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._art_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists, p_gens):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        if self._pre_trained_embedding == []:
          print('not get pre')
          embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        else:
          print('get pre')
          embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=tf.constant_initializer(np.array(self._pre_trained_embedding)), trainable=True)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_rqs_inputs = tf.nn.embedding_lookup(embedding, self._rqs_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        # emb_rqs_inputs = tf.Print(emb_rqs_inputs, [tf.reduce_sum(emb_rqs_inputs, axis=(1,2))])
        emb_art_inputs = tf.nn.embedding_lookup(embedding, self._art_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)

        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      rqs_outputs, fw_st_rqs, bw_st_rqs = self._add_encoder(emb_rqs_inputs, self._rqs_lens, 'rqs')
      art_outputs, fw_st_art, bw_st_art = self._add_encoder(emb_art_inputs, self._art_lens, 'art')
      self._rqs_states = rqs_outputs
      self._art_states = tf.unstack(art_outputs, axis=1) # list length max_rqs_seq_len (batch_size, hidden_dims)
      emb_enc_inputs = query_aware_attention(self._rqs_states, self._art_states, self._rqs_padding_mask)

      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._art_lens, 'enc')
      self._enc_states = enc_outputs


        # self._logits = tf.Print(self._logits, [self._logits, self._logits.shape])
      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)

      # Add the predictor.
      with tf.variable_scope('predictor'):
        self._logits, self._y_pred_cls = self._add_predictor(tf.reduce_mean(self._enc_states, axis=1))
        # self._logits, self._y_pred_cls = self._add_predictor(self._dec_in_state.c)


      # Add the decoder1.
      with tf.variable_scope('decoder1'):
        decoder_outputs_1, self._dec_out_state_1, self.attn_dists_1, self.p_gens_1, self.coverage_1 = self._add_decoder(emb_dec_inputs)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection1'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores_1 = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs_1):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores_1.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists_1 = [tf.nn.softmax(s) for s in vocab_scores_1] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists_1 = self._calc_final_dist(vocab_dists_1, self.attn_dists_1, self.p_gens_1)
      else: # final distribution is just vocabulary distribution
        final_dists_1 = vocab_dists_1

      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss_predictor'):
          pred_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=tf.one_hot(indices = tf.cast(self._rst_batch, dtype=tf.uint8), depth=2))
          self._pred_loss = tf.reduce_mean(pred_cross_entropy)

        with tf.variable_scope('loss_1'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step_1 = [] # will be list length max_dec_steps containing shape (batch_size)

            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists_1):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)
              loss_per_step_1.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss_1 = _mask_and_avg(loss_per_step_1, self._dec_padding_mask, self._rst_batch)
          else: # baseline model
            self._loss_1 = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores_1, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss_1', self._loss_1)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss_1'):
              self._coverage_loss_1 = _coverage_loss(self.attn_dists_1, self._dec_padding_mask, self._rst_batch)
              tf.summary.scalar('coverage_loss_1', self._coverage_loss_1)
            self._total_loss_1 = self._loss_1 + hps.cov_loss_wt * self._coverage_loss_1
            tf.summary.scalar('total_loss_1', self._total_loss_1)

      # Add the decoder2.
      with tf.variable_scope('decoder2'):
        decoder_outputs_2, self._dec_out_state_2, self.attn_dists_2, self.p_gens_2, self.coverage_2 = self._add_decoder(emb_dec_inputs)

      with tf.variable_scope('output_projection2'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores_2 = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs_2):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores_2.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists_2 = [tf.nn.softmax(s) for s in vocab_scores_2] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists_2 = self._calc_final_dist(vocab_dists_2, self.attn_dists_2, self.p_gens_2)
      else: # final distribution is just vocabulary distribution
        final_dists_2 = vocab_dists_2

      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss_2'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step_2 = [] # will be list length max_dec_steps containing shape (batch_size)

            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists_2):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)
              loss_per_step_2.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss_2 = _mask_and_avg(loss_per_step_2, self._dec_padding_mask, 1 - self._rst_batch)
          else: # baseline model
            self._loss_2 = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores_2, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss_2', self._loss_2)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss_2'):
              # print(1 - self._rst_batch)
              self._coverage_loss_2 = _coverage_loss(self.attn_dists_2, self._dec_padding_mask, 1 - self._rst_batch)
              tf.summary.scalar('coverage_loss_2', self._coverage_loss_2)
            self._total_loss_2 = self._loss_2 + hps.cov_loss_wt * self._coverage_loss_2
            tf.summary.scalar('total_loss_2', self._total_loss_2)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists_1)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      assert len(final_dists_2)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)


      final_dists_1 = final_dists_1[0]
      topk_probs_1, self._topk_ids_1 = tf.nn.top_k(final_dists_1, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs_1 = tf.log(topk_probs_1)
      final_dists_2 = final_dists_2[0]
      topk_probs_2, self._topk_ids_2 = tf.nn.top_k(final_dists_2, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs_2 = tf.log(topk_probs_2)


  def _add_pred_train_op(self):
    pred_loss_to_minimize = self._pred_loss
    tvars_pred = tf.trainable_variables()
    gradients_pred = tf.gradients(pred_loss_to_minimize, tvars_pred, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    # with tf.device("/gpu:3"):
    grads_pred, global_norm_pred = tf.clip_by_global_norm(gradients_pred, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm_pred', global_norm_pred)

    # Apply adagrad optimizer
    optimizer_pred = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    # with tf.device("/gpu:3"):
    self._train_op_pred = optimizer_pred.apply_gradients(zip(grads_pred, tvars_pred), global_step=self.global_step, name='train_step_pred')
    # self._train_op_pred = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self._pred_loss)

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss_1+self._total_loss_2+0.1*self._pred_loss if self._hps.coverage else self._loss_1+self._loss_2+0.1*self._pred_loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    # with tf.device("/gpu:3"):
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    # with tf.device("/gpu:3"):
    self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step_1')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    # with tf.device("/gpu:3"):
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if self._hps.mode == 'train':
      # self._add_pred_train_op()
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
      'loss_pred': self._pred_loss,
      'encoder_output': self._enc_states,
        'summaries': self._summaries,
        'loss_1': self._loss_1,
        'global_step': self.global_step,
        'train_op': self._train_op,
        'loss_2': self._loss_2,
    }
    if self._hps.coverage:
      to_return['coverage_loss_1'] = self._coverage_loss_1
      to_return['coverage_loss_2'] = self._coverage_loss_2
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'loss_pred': self._pred_loss,
        'summaries': self._summaries,
        'loss_1': self._loss_1,
        'global_step': self.global_step,
        'loss_2': self._loss_2,
    }
    if self._hps.coverage:
      to_return['coverage_loss_1'] = self._coverage_loss_1
      to_return['coverage_loss_2'] = self._coverage_loss_2
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._art_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self._rst_batch: batch.rst_batch
    }
    to_return = {'pred': self._y_pred_cls}
    pred_results = sess.run(to_return, feed_dict=feed)


    if pred_results['pred'][0] == 1:
      to_return = {
        "ids": self._topk_ids_1,
        "probs": self._topk_log_probs_1,
        "states": self._dec_out_state_1,
        "attn_dists": self.attn_dists_1,
      }

      if FLAGS.pointer_gen:
        feed[self._enc_batch_extend_vocab] = batch.art_batch_extend_vocab
        feed[self._max_art_oovs] = batch.max_art_oovs
        to_return['p_gens'] = self.p_gens_1

      if self._hps.coverage:
        feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
        to_return['coverage'] = self.coverage_1
    else:
      to_return = {
        "ids": self._topk_ids_2,
        "probs": self._topk_log_probs_2,
        "states": self._dec_out_state_2,
        "attn_dists": self.attn_dists_2,
      }

      if FLAGS.pointer_gen:
        feed[self._enc_batch_extend_vocab] = batch.art_batch_extend_vocab
        feed[self._max_art_oovs] = batch.max_art_oovs
        to_return['p_gens'] = self.p_gens_2

      if self._hps.coverage:
        feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
        to_return['coverage'] = self.coverage_2


    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]

    return pred_results['pred'][0], results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask, batch_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """
  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  values_per_ex = tf.expand_dims(values_per_ex, 1)
  values_per_ex = batch_mask * values_per_ex

  nums = tf.reduce_sum(batch_mask)

  avg = tf.reduce_sum(values_per_ex)/nums

  return avg



def _coverage_loss(attn_dists, padding_mask, batch_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask, batch_mask)
  return coverage_loss
