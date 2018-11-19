"""
LSTM model for hate-detector
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import numpy as np
import tensorflow as tf
from types import SimpleNamespace
TF_INT = tf.int32
TF_FLOAT = tf.float32
NP_FLOAT = np.float32


class ZeroPaddedSequence(object):
    '''placeholder which expects a zero padded sequence batch
        If time_major true, the sequence is expected to be shaped
        [max_time, batch_size, depth].
        If time_major false, the sequence is expected to be shaped
        [batch_size, max_time, depth].
    '''
    def _define_inputs(self, num_inputs, batch_size, time_major):
        shape = ([None, batch_size, num_inputs] if time_major
                 else [batch_size, None, num_inputs])
        inputs = tf.placeholder(dtype=TF_FLOAT, shape=shape)
        return inputs

    def __init__(self, inputs=None, num_inputs=None, time_major=False,
                 batch_size=None):
        if inputs is None:
            inputs = self._define_inputs(num_inputs=num_inputs,
                                         batch_size=batch_size,
                                         time_major=time_major)
        self.inputs = inputs
        self.time_major = time_major
        with tf.name_scope('padded_input'):
            self.value = inputs
            self.mask = tf.reduce_sum(self.value, axis=-1)
            time_axis = (0 if self.time_major else 1)
            self.sequence_length = tf.reduce_sum(
                tf.cast(tf.greater(self.mask, 0), TF_INT),
                axis=time_axis)


class LstmModel(object):
    def _define_rnn(self, num_units, inputs=None, num_inputs=None,
                    batch_size=None, keep_prob=None):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        if keep_prob is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=cell, input_keep_prob=keep_prob)
        inputs = ZeroPaddedSequence(
            inputs=inputs,  num_inputs=num_inputs,
            time_major=self.time_major,  batch_size=batch_size)
        batch_dim = (1 if self.time_major else 0)
        if inputs.value.shape[batch_dim].value is not None:
            batch_size = inputs.value.shape[batch_dim].value
        else:
            batch_size = tf.shape(inputs.value)[batch_dim]
        state_initial = cell.zero_state(batch_size, dtype=TF_FLOAT)
        outputs, state_final = tf.nn.dynamic_rnn(
            cell, inputs=inputs.value,
            initial_state=state_initial,
            sequence_length=inputs.sequence_length,
            time_major=self.time_major)
        return SimpleNamespace(
            cell=cell,
            outputs=outputs,
            inputs=inputs,
            state_initial=state_initial,
            state_final=state_final
        )

    def _define_classifier(self, num_outputs, keep_prob=None):
        time_dim = (0 if self.time_major else 1)
        inputs = (
            tf.reduce_sum(self.rnn.outputs, axis=time_dim) /
            tf.cast(self.rnn.inputs.sequence_length,
                    TF_FLOAT)[..., tf.newaxis])
        # model = tf.keras.models.Sequential()
        # if keep_prob is not None:
        #     model.add(tf.layers.Dropout(rate=1-keep_prob))
        # model.add(tf.layers.Dense(units=num_outputs))
        model = tf.layers.Dense(units=num_outputs)
        x = inputs
        if keep_prob is not None:
            x = tf.nn.dropout(x, keep_prob=keep_prob)
        logits = model(x)
        return SimpleNamespace(
            model=model,
            inputs=inputs,
            logits=logits,
            predictions=tf.nn.sigmoid(logits)
        )

    def get_regularizer(self, coef=0.001):
        models = [self.rnn.cell, self.classifier.model]
        weights = set()
        for model in models:
            weights.update(set([var for var in model.weights
                                if 'bias' not in var.name]))

        regularizer = tf.keras.regularizers.l2(coef)
        return tf.add_n([regularizer(w) for w in weights])

    def __init__(self, num_units, inputs=None, num_inputs=None,
                 num_outputs=1,
                 keep_prob=None,
                 batch_size=None, time_major=True):
        ''' inputs: zero padded one-hot encoded sequences '''
        self.time_major = time_major
        self.batch_size = batch_size
        if keep_prob is None:
            keep_prob = SimpleNamespace(rnn=None, classifier=None)
        # Rnn model
        self.rnn = self._define_rnn(num_units=num_units,
                                    inputs=inputs,
                                    num_inputs=num_inputs,
                                    batch_size=batch_size,
                                    keep_prob=keep_prob.rnn)
        # Classifier definition
        self.classifier = self._define_classifier(
            num_outputs=num_outputs,
            keep_prob=keep_prob.classifier)


class LstmEstimator(object):
    @staticmethod
    def model_fn(features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            keep_prob = params['keep_prob']
        else:
            keep_prob = None

        model = LstmModel(inputs=features,
                          num_units=params['num_units'],
                          keep_prob=keep_prob,
                          time_major=params['time_major'])
        predicted_classes = tf.cast(model.classifier.predictions > 0.5,
                                    TF_INT)
        # Predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes,
                'probabilities': model.classifier.predictions,
                'logits': model.classifier.predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        # Train
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=model.classifier.logits)

        if params['regularizer'] is not None:
            loss = loss + model.get_regularizer(params['regularizer'])

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op,
            eval_metric_ops=metrics)

    @staticmethod
    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, 1)
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        dataset = dataset.map(lambda x, y: (tf.cast(x, TF_FLOAT),
                                            tf.cast(y, TF_INT)))
        return dataset

    @staticmethod
    def eval_input_fn(features, labels, batch_size):
        """An input function for evaluation or prediction"""
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, 1)
        inputs = (features if labels is None
                  else (features, labels))
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)
        if labels is None:
            dataset = dataset.map(lambda x: tf.cast(x, TF_FLOAT))
        else:
            dataset = dataset.map(lambda x, y: (tf.cast(x, TF_FLOAT),
                                                tf.cast(y, TF_INT)))
        return dataset

    def __init__(self, num_inputs, num_units, batch_size=32,
                 model_dir=None):
        ''' '''
        self.time_major = False
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={'num_units': num_units,
                    'time_major': self.time_major,
                    'keep_prob': SimpleNamespace(
                        rnn=0.5,
                        classifier=0.9),
                    'regularizer': 0.0001},
            model_dir=model_dir
            )

    def fit(self, x, y, steps=1000):
        self.estimator.train(
            input_fn=lambda: self.train_input_fn(x, y, self.batch_size),
            steps=steps
            )

    def evaluate(self, x, y):
        eval_result = self.estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(x, y, self.batch_size))
        return eval_result
