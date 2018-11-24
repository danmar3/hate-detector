"""
LSTM model for hate-detector
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import collections
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
            _mask = tf.reduce_sum(tf.abs(self.value), axis=-1)
            self.mask = tf.cast(tf.greater(_mask, 0), TF_INT)
            time_axis = (0 if self.time_major else 1)
            self.sequence_length = tf.reduce_sum(self.mask, axis=time_axis)


class BidirectionalCell(object):
    def __init__(self, num_units):
        self.forward = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
        self.backward = tf.nn.rnn_cell.LSTMCell(num_units=num_units)

    def weights(self):
        return self.forward.weights + self.backward.weights


class StackedCell(object):
    def __init__(self, cells):
        self.cells = cells

    def weights(self):
        weights = list()
        for cell in self.cells:
            weights.extend(cell.weights)
        return weights


class LstmModel(object):
    def _define_cell(self, num_units, keep_prob):
        cells = list()
        for i in range(len(num_units)):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units[i])
            if keep_prob[i] is not None:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=cell, input_keep_prob=keep_prob[i])
            cells.append(cell)
        if len(cells) == 1:
            cell = cells[0]
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def _run_rnn(self, inputs, batch_size):
        state_initial = self.cell.zero_state(batch_size, dtype=TF_FLOAT)
        outputs, state_final = tf.nn.dynamic_rnn(
            self.cell, inputs=inputs.value,
            initial_state=state_initial,
            sequence_length=inputs.sequence_length,
            time_major=self.time_major)
        return state_initial, state_final, outputs

    def _define_rnn(self, num_units, inputs=None, num_inputs=None,
                    batch_size=None, keep_prob=None):
        self.cell = self._define_cell(num_units=num_units, keep_prob=keep_prob)
        inputs = ZeroPaddedSequence(
            inputs=inputs,  num_inputs=num_inputs,
            time_major=self.time_major,  batch_size=batch_size)
        batch_dim = (1 if self.time_major else 0)
        if inputs.value.shape[batch_dim].value is not None:
            batch_size = inputs.value.shape[batch_dim].value
        else:
            batch_size = tf.shape(inputs.value)[batch_dim]
        state_initial, state_final, outputs = \
            self._run_rnn(inputs=inputs, batch_size=batch_size)
        return SimpleNamespace(
            cell=self.cell,
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

    def fit_loss(self, labels):
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=self.classifier.logits)
        return loss

    def __init__(self, num_units, inputs=None, num_inputs=None,
                 num_outputs=1,
                 keep_prob=None,
                 batch_size=None, time_major=True):
        ''' inputs: zero padded one-hot encoded sequences '''
        self.time_major = time_major
        self.batch_size = batch_size
        if not isinstance(num_units, collections.Iterable):
            num_units = [num_units]
        if keep_prob is None:
            keep_prob = SimpleNamespace(rnn=None, classifier=None)
        if not isinstance(keep_prob.rnn, collections.Iterable):
            keep_prob.rnn = [keep_prob.rnn]*len(num_units)
        # if not isinstance(keep_prob.classifier, collections.iterable):
        #     keep_prob.classifier = [keep_prob.classifier]*len(num_units)
        assert len(num_units) == len(keep_prob.rnn),\
            'num_inputs and keep_prob.rnn must have same length'
        # assert len(num_inputs) == len(keep_prob.classifier),\
        #     'num_inputs and keep_prob.classifier must have same length'
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


class BiLstmModel(LstmModel):
    def _define_cell(self, num_units, keep_prob):
        # assert len(num_units) == 1, 'not implemented'
        cells = [BidirectionalCell(num_units=units)
                 for i, units in enumerate(num_units)]
        return StackedCell(cells=cells)

    def _run_rnn(self, inputs, batch_size):
        state_initial = list()
        state_final = list()
        sequence_length = inputs.sequence_length
        inputs = inputs.value
        for i, cell_i in enumerate(self.cell.cells):
            state_initial.append(
                (cell_i.forward.zero_state(batch_size, dtype=TF_FLOAT),
                 cell_i.backward.zero_state(batch_size, dtype=TF_FLOAT))
                                 )
            outputs, state_final = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_i.forward,
                cell_bw=cell_i.backward,
                inputs=inputs,
                initial_state_fw=state_initial[-1][0],
                initial_state_bw=state_initial[-1][1],
                sequence_length=sequence_length,
                time_major=self.time_major,
                scope='rnnlayer_{}'.format(i)
                )
            outputs = tf.concat(outputs, axis=-1)
            inputs = outputs
        return state_initial, state_final, outputs


class AggreagtedLstm(LstmModel):
    def _define_classifier(self, num_outputs, keep_prob=None):
        time_axis = (0 if self.time_major else 1)
        inputs = self.rnn.outputs
        # model = tf.keras.models.Sequential()
        # if keep_prob is not None:
        #     model.add(tf.layers.Dropout(rate=1-keep_prob))
        # model.add(tf.layers.Dense(units=num_outputs))
        dense = tf.layers.Dense(units=num_outputs)
        x = inputs
        if keep_prob is not None:
            x = tf.nn.dropout(x, keep_prob=keep_prob)
        #
        # x_ta = tf.TensorArray(dtype=x.dtype, size=tf.shape(x)[time_axis])
        # x_ta.unstack()  # TODO)
        logits = dense(x)
        sentence_mask = tf.cast(self.rnn.inputs.mask[..., tf.newaxis],
                                TF_FLOAT)
        predictions = tf.reduce_sum(
            tf.nn.sigmoid(logits)*sentence_mask,
            axis=time_axis)
        predictions = predictions/tf.cast(self.rnn.inputs.sequence_length,
                                          predictions.dtype)[..., tf.newaxis]
        return SimpleNamespace(
            model=dense,
            inputs=inputs,
            logits=logits,
            predictions=predictions
        )

    def fit_loss(self, labels):
        if self.time_major:
            time_axis = 0
            labels = labels[tf.newaxis, ...]
            labels = tf.tile(
                labels, [tf.shape(self.rnn.inputs.value)[time_axis], 1, 1])
        else:
            time_axis = 1
            labels = labels[:, tf.newaxis, ...]
            labels = tf.tile(
                labels, [1, tf.shape(self.rnn.inputs.value)[time_axis], 1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, TF_FLOAT),
            logits=self.classifier.logits)
        loss = tf.reduce_sum(loss, tuple(range(2, loss.shape.ndims)))
        loss = tf.reduce_sum(loss * tf.cast(self.rnn.inputs.mask, loss.dtype),
                             axis=time_axis)
        loss = tf.reduce_mean(
            loss / tf.cast(self.rnn.inputs.sequence_length,
                           loss.dtype))
        return loss


class AggreagtedBiLstm(AggreagtedLstm, BiLstmModel):
    pass


class LstmEstimator(object):
    MlModel = LstmModel

    def model_fn(self, features, labels, mode, params):
        features = features['x']
        if mode == tf.estimator.ModeKeys.TRAIN:
            keep_prob = params['keep_prob']
        else:
            keep_prob = None

        model = self.MlModel(inputs=features,
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
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        # Train
        loss = model.fit_loss(labels=labels)

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

        optimizer = tf.train.AdamOptimizer(
            learning_rate=(0.01 if params['learning_rate'] is None
                           else params['learning_rate']))
        if params['gradient_clip'] is None:
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step())
        else:
            grads_and_vars = optimizer.compute_gradients(loss)
            gradients, v = zip(*grads_and_vars)
            # 3. process the gradients
            gradients, _ = tf.clip_by_global_norm(
                gradients, params['gradient_clip'])  # 1.25 #0.025 #0.001(last used)
            # 4. apply the gradients to the optimization procedure
            train_op = optimizer.apply_gradients(
                zip(gradients, v),
                global_step=tf.train.get_global_step())
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
        dataset = dataset.shuffle(500).repeat().batch(batch_size)
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

    def __init__(self, num_inputs, num_units, keep_prob=None,
                 regularizer=0.0001, learning_rate=0.001,
                 gradient_clip=None,
                 batch_size=32,
                 model_dir=None):
        ''' '''
        if keep_prob is None:
            keep_prob = SimpleNamespace(rnn=0.5, classifier=0.9)
        self.time_major = False
        self.batch_size = batch_size
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={'num_units': num_units,
                    'time_major': self.time_major,
                    'keep_prob': keep_prob,
                    'regularizer': regularizer,
                    'learning_rate': learning_rate,
                    'gradient_clip': gradient_clip},
            model_dir=model_dir
            )

    def fit(self, x, y, steps=1000):
        # self.estimator.train(
        #     input_fn=lambda: self.train_input_fn(x, y, self.batch_size),
        #     steps=steps
        #    )
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x},
            y=y, batch_size=self.batch_size,
            num_epochs=None,
            shuffle=True)
        self.estimator.train(
            input_fn=train_input_fn,
            steps=steps
            )

    def evaluate(self, x, y):
        # eval_result = self.estimator.evaluate(
        #     input_fn=lambda: self.eval_input_fn(x, y, self.batch_size))
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x}, y=y,
            num_epochs=1,
            shuffle=False)
        eval_result = self.estimator.evaluate(
            input_fn=eval_input_fn)
        return eval_result

    def predict(self, x):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x}, y=None,
            num_epochs=1,
            shuffle=False)
        eval_result = self.estimator.predict(
            input_fn=eval_input_fn)
        return eval_result


class BiLstmEstimator(LstmEstimator):
    MlModel = BiLstmModel


class AggregatedLstmEstimator(LstmEstimator):
    MlModel = AggreagtedLstm


class AggregatedBiLstmEstimator(LstmEstimator):
    MlModel = AggreagtedBiLstm
