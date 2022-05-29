# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.compat.v1.nn import dynamic_rnn
from recommenders.models.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["DIN_RECModel"]


class DIN_RECModel(SequentialBaseModel):
    """Deep Interest model

    :Citation:

    """

    def _build_seq_graph(self):
        """The main function to create din model.

        Returns:
            object: the output of din section.
        """
        # hparams = self.hparams

        with tf.compat.v1.variable_scope("din"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2 # [batch_size, seq, embed]
            )
            self.mask = self.iterator.mask

            query = tf.expand_dims(self.target_item_embedding, axis=1) # [batch_size, 1, embed]
            self.weighted_hist_input = self._target_attention(query, hist_input, hist_input)

            user_embed = self.weighted_hist_input
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.compat.v1.summary.histogram("model_output", model_output)
            return model_output

    def _target_attention(self, query, key, value):
        
        with tf.compat.v1.variable_scope("target_attention"):
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_logits = tf.math.reduce_sum(
                tf.multiply(query, key), axis=-1
            ) # [batch_size, sequence]
            mask_paddings = tf.ones_like(attention_logits) * (-(2 ** 32) + 1)
            attention_weights = tf.math.softmax(
                tf.where(boolean_mask, attention_logits, mask_paddings), axis=-1
            ) #[batch_size, sequence]
            weighted_sum = tf.reduce_sum(
                tf.multiply(tf.expand_dims(attention_weights, -1), value), axis=-2
            ) # [batch_size, emb]

        return weighted_sum