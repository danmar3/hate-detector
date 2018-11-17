"""
LSTM model for hate-detector
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import tensorflow as tf
from types import SimpleNamespace
TF_INT = tf.int32
TF_FLOAT = tf.float32


class ZeroPaddedSequence(object):
    '''placeholder which expects a zero padded sequence batch
        If time_major true, the sequence is expected to be shaped
        [max_time, batch_size, depth].
        If time_major false, the sequence is expected to be shaped
        [batch_size, max_time, depth].
    '''
    def __init__(self, inputs_size, time_major=False):
        self.time_major = time_major
        with tf.name_scope('padded_input'):
            self.value = tf.placeholder(
                dtype=TF_FLOAT,
                shape=[None, None, inputs_size])
            self.mask = tf.reduce_sum(self.value, axis=-1)
            time_axis = (0 if self.time_major else 1)
            self.sequence_length = tf.reduce_sum(
                tf.cast(tf.greater(self.mask, 0), TF_INT),
                axis=time_axis)


class LstmModel(object):

    def __init__(self, num_inputs, num_units,
                 batch_size=None, time_major=True):
        self.time_major = time_major
        self.batch_size = batch_size
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        inputs = ZeroPaddedSequence(inputs_size=num_inputs,
                                    time_major=time_major)
        if batch_size is None:
            batch_dim = (1 if self.time_major else 0)
            batch_size = tf.shape(inputs.value)[batch_dim]
        self.state_initial = self.cell.zero_state(batch_size, dtype=TF_FLOAT)
        outputs, state_final = tf.nn.dynamic_rnn(
            self.cell, inputs=inputs.value,
            initial_state=self.state_initial,
            sequence_length=inputs.sequence_length,
            time_major=time_major)
        self.rnn = SimpleNamespace(
            outputs=outputs,
            inputs=inputs,
            state_final=state_final,
        )
        # Classifier definition
        time_dim = (0 if self.time_major else 1)
        classifier_inputs = (
            tf.reduce_sum(self.rnn.outputs, axis=time_dim) /
            tf.cast(inputs.sequence_length, TF_FLOAT)[..., tf.newaxis])
        self.classifier = SimpleNamespace(
            # model=classifier_model,
            inputs=classifier_inputs
        )

    def predict(self, inputs):
        pass
