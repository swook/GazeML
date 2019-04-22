"""Deep Pictorial Gaze architecture."""
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf

from core import BaseDataSource, BaseModel
from datasources import UnityEyes
import util.gaze


class DPG(BaseModel):
    """Deep Pictorial Gaze architecture as introduced in [Park et al. ECCV'18]."""

    def __init__(self, tensorflow_session=None, first_layer_stride=2, num_modules=3,
                 num_feature_maps=32, growth_rate=8, extra_tags=[], **kwargs):
        """Specify DPG-specific parameters."""
        self._hg_first_layer_stride = first_layer_stride
        self._hg_num_modules = num_modules
        self._hg_num_feature_maps= num_feature_maps
        self._dn_growth_rate = growth_rate
        self._extra_tags = extra_tags

        # Call parent class constructor
        super().__init__(tensorflow_session, **kwargs)

    _hg_first_layer_stride = 2
    _hg_num_modules = 3
    _hg_num_feature_maps = 32
    _hg_num_residual_blocks = 1
    _hg_num_gazemaps = 2

    _dn_growth_rate = 8
    _dn_compression_factor = 0.5
    _dn_num_layers_per_block = (4, 4, 4, 4)
    _dn_num_dense_blocks = len(_dn_num_layers_per_block)

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        first_data_source = next(iter(self._train_data.values()))
        input_tensors = first_data_source.output_tensors
        if self._data_format == 'NHWC':
            _, eh, ew, _ = input_tensors['eye'].shape.as_list()
        else:
            _, _, eh, ew = input_tensors['eye'].shape.as_list()
        return 'DPG_i%dx%d_f%dx%d_n%d_m%d_k%d_%s' % (
            ew, eh,
            int(ew / self._hg_first_layer_stride),
            int(eh / self._hg_first_layer_stride),
            self._hg_num_feature_maps, self._hg_num_modules,
            self._dn_growth_rate,
            '-'.join(self._extra_tags) if len(self._extra_tags) > 0 else '',
        )

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        # Step learning rate decay
        multiplier = np.power(0.1, int(current_step / 10000))
        self._tensorflow_session.run(self.assign_learning_rate_multiplier, feed_dict={
            self.learning_rate_multiplier_placeholder: multiplier,
        })

    _column_of_ones = None
    _column_of_zeros = None

    def _augment_training_images(self, images, mode):
        if mode == 'test':
            return images
        with tf.variable_scope('augment'):
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 2, 3, 1])
            n, h, w, _ = images.shape.as_list()
            if self._column_of_ones is None:
                self._column_of_ones = tf.ones((n, 1))
                self._column_of_zeros = tf.zeros((n, 1))
            transforms = tf.concat([
                self._column_of_ones,
                self._column_of_zeros,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*w),
                self._column_of_zeros,
                self._column_of_ones,
                tf.truncated_normal((n, 1), mean=0, stddev=.05*h),
                self._column_of_zeros,
                self._column_of_zeros,
            ], axis=1)
            images = tf.contrib.image.transform(images, transforms, interpolation='BILINEAR')
            if self._data_format == 'NCHW':
                images = tf.transpose(images, perm=[0, 3, 1, 2])
        return images

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y1 = input_tensors['gazemaps'] if 'gazemaps' in input_tensors else None
        y2 = input_tensors['gaze'] if 'gaze' in input_tensors else None

        with tf.variable_scope('input_data'):
            # self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)
            if y1 is not None:
                self.summary.feature_maps('gazemaps', y1, data_format=self._data_format_longer)

        outputs = {}
        loss_terms = {}
        metrics = {}

        # Lightly augment training data
        x = self._augment_training_images(x, mode)

        with tf.variable_scope('hourglass'):
            # Prepare for Hourglass by downscaling via conv
            with tf.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(x, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

            # Hourglass blocks
            x_prev = x
            gmap = None
            for i in range(self._hg_num_modules):
                with tf.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, gmap = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    x_prev = x
            if y1 is not None:
                # Cross-entropy loss
                metrics['gazemaps_ce'] = -tf.reduce_mean(tf.reduce_sum(
                    y1 * tf.log(tf.clip_by_value(gmap, 1e-10, 1.0)),  # avoid NaN
                    axis=[1, 2, 3]))
                # metrics['gazemaps_ce'] = tf.losses.softmax_cross_entropy(
                #     tf.reshape(y1, (self._batch_size, -1)),
                #     tf.reshape(gmap, (self._batch_size, -1)),
                #     loss_collection=None,
                # )
            x = gmap
            outputs['gazemaps'] = gmap
            self.summary.feature_maps('bottleneck', gmap, data_format=self._data_format_longer)

        with tf.variable_scope('densenet'):
            # DenseNet blocks to regress to gaze
            for i in range(self._dn_num_dense_blocks):
                with tf.variable_scope('block%d' % (i + 1)):
                    x = self._apply_dense_block(x,
                                                num_layers=self._dn_num_layers_per_block[i])
                    if i == self._dn_num_dense_blocks - 1:
                        break
                with tf.variable_scope('trans%d' % (i + 1)):
                    x = self._apply_transition_layer(x)

            # Global average pooling
            with tf.variable_scope('post'):
                x = self._apply_bn(x)
                x = tf.nn.relu(x)
                if self._data_format == 'NCHW':
                    x = tf.reduce_mean(x, axis=[2, 3])
                else:
                    x = tf.reduce_mean(x, axis=[1, 2])
                x = tf.contrib.layers.flatten(x)

            # Output layer
            with tf.variable_scope('output'):
                x = self._apply_fc(x, 2)
                outputs['gaze'] = x
                if y2 is not None:
                    metrics['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y2))
                    metrics['gaze_ang'] = util.gaze.tensorflow_angular_error_from_pitchyaw(y2, x)

        # Combine two loss terms
        if y1 is not None and y2 is not None:
            loss_terms['combined_loss'] = 1e-5*metrics['gazemaps_ce'] + metrics['gaze_mse']

        # Define outputs
        return outputs, loss_terms, metrics

    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x

    def _build_hourglass(self, x, steps_to_go, num_features, depth=1):
        with tf.variable_scope('depth%d' % depth):
            # Upper branch
            up1 = x
            for i in range(self._hg_num_residual_blocks):
                up1 = self._build_residual_block(up1, num_features, num_features,
                                                 name='up1_%d' % (i + 1))
            # Lower branch
            low1 = self._apply_pool(x, kernel_size=2, stride=2)
            for i in range(self._hg_num_residual_blocks):
                low1 = self._build_residual_block(low1, num_features, num_features,
                                                  name='low1_%d' % (i + 1))
            # Recursive
            low2 = None
            if steps_to_go > 1:
                low2 = self._build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
            else:
                low2 = low1
                for i in range(self._hg_num_residual_blocks):
                    low2 = self._build_residual_block(low2, num_features, num_features,
                                                      name='low2_%d' % (i + 1))
            # Additional residual blocks
            low3 = low2
            for i in range(self._hg_num_residual_blocks):
                low3 = self._build_residual_block(low3, num_features, num_features,
                                                  name='low3_%d' % (i + 1))
            # Upsample
            if self._data_format == 'NCHW':  # convert to NHWC
                low3 = tf.transpose(low3, (0, 2, 3, 1))
            up2 = tf.image.resize_bilinear(
                    low3,
                    up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                    align_corners=True,
                  )
            if self._data_format == 'NCHW':  # convert back from NHWC
                up2 = tf.transpose(up2, (0, 3, 1, 2))

        return up1 + up2

    def _build_hourglass_after(self, x_prev, x_now, do_merge=True):
        with tf.variable_scope('after'):
            for j in range(self._hg_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._hg_num_feature_maps,
                                                   self._hg_num_feature_maps,
                                                   name='after_hg_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.variable_scope('gmap'):
                gmap = self._apply_conv(x_now, self._hg_num_gazemaps, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.variable_scope('merge'):
                with tf.variable_scope('gmap'):
                    x_gmaps = self._apply_conv(gmap, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_gmaps

        # Perform softmax on gazemaps
        if self._data_format == 'NCHW':
            n, c, h, w = gmap.shape.as_list()
            gmap = tf.reshape(gmap, (n, -1))
            gmap = tf.nn.softmax(gmap)
            gmap = tf.reshape(gmap, (n, c, h, w))
        else:
            n, h, w, c = gmap.shape.as_list()
            gmap = tf.transpose(gmap, perm=[0, 3, 1, 2])
            gmap = tf.reshape(gmap, (n, -1))
            gmap = tf.nn.softmax(gmap)
            gmap = tf.reshape(gmap, (n, c, h, w))
            gmap = tf.transpose(gmap, perm=[0, 2, 3, 1])
        return x_next, gmap

    def _apply_dense_block(self, x, num_layers):
        assert isinstance(num_layers, int) and num_layers > 0
        c_index = 1 if self._data_format == 'NCHW' else 3
        x_prev = x
        for i in range(num_layers):
            with tf.variable_scope('layer%d' % (i + 1)):
                n = x.shape.as_list()[c_index]
                with tf.variable_scope('bottleneck'):
                    x = self._apply_composite_function(x,
                                                       num_features=min(n, 4*self._dn_growth_rate),
                                                       kernel_size=1)
                with tf.variable_scope('composite'):
                    x = self._apply_composite_function(x, num_features=self._dn_growth_rate,
                                                       kernel_size=3)
                if self._data_format == 'NCHW':
                    x = tf.concat([x, x_prev], axis=1)
                else:
                    x = tf.concat([x, x_prev], axis=-1)
                x_prev = x
        return x

    def _apply_transition_layer(self, x):
        c_index = 1 if self._data_format == 'NCHW' else 3
        x = self._apply_composite_function(
            x, num_features=int(self._dn_compression_factor * x.shape.as_list()[c_index]),
            kernel_size=1)
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid',
                                        data_format=self._data_format_longer)
        return x

    def _apply_composite_function(self, x, num_features=_dn_growth_rate, kernel_size=3):
        x = self._apply_bn(x)
        x = tf.nn.relu(x)
        x = self._apply_conv(x, num_features=num_features, kernel_size=kernel_size, stride=1)
        return x
