"""ELG architecture."""
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf

from core import BaseDataSource, BaseModel


def _tf_mse(x, y):
    """Tensorflow call for mean-squared error."""
    return tf.reduce_mean(tf.squared_difference(x, y))


class ELG(BaseModel):
    """ELG architecture as introduced in [Park et al. ETRA'18]."""

    def __init__(self, tensorflow_session=None, first_layer_stride=1,
                 num_modules=2, num_feature_maps=32, **kwargs):
        """Specify ELG-specific parameters."""
        self._hg_first_layer_stride = first_layer_stride
        self._hg_num_modules = num_modules
        self._hg_num_feature_maps= num_feature_maps

        # Call parent class constructor
        super().__init__(tensorflow_session, **kwargs)

    _hg_first_layer_stride = 1
    _hg_num_modules = 2
    _hg_num_feature_maps = 32
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        first_data_source = next(iter(self._train_data.values()))
        input_tensors = first_data_source.output_tensors
        if self._data_format == 'NHWC':
            _, eh, ew, _ = input_tensors['eye'].shape.as_list()
        else:
            _, _, eh, ew = input_tensors['eye'].shape.as_list()
        return 'ELG_i%dx%d_f%dx%d_n%d_m%d' % (
            ew, eh,
            int(ew / self._hg_first_layer_stride),
            int(eh / self._hg_first_layer_stride),
            self._hg_num_feature_maps, self._hg_num_modules,
        )

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        # Set difficulty of training data
        data_source = next(iter(self._train_data.values()))
        data_source.set_difficulty(min((1. / 1e6) * current_step, 1.))

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y1 = input_tensors['heatmaps'] if 'heatmaps' in input_tensors else None
        y2 = input_tensors['landmarks'] if 'landmarks' in input_tensors else None
        y3 = input_tensors['radius'] if 'radius' in input_tensors else None

        with tf.variable_scope('input_data'):
            self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)
            if y1 is not None:
                self.summary.feature_maps('hmaps_true', y1, data_format=self._data_format_longer)

        outputs = {}
        loss_terms = {}
        metrics = {}

        with tf.variable_scope('hourglass'):
            # TODO: Find better way to specify no. landmarks
            if y1 is not None:
                if self._data_format == 'NCHW':
                    self._hg_num_landmarks = y1.shape.as_list()[1]
                if self._data_format == 'NHWC':
                    self._hg_num_landmarks = y1.shape.as_list()[3]
            else:
                self._hg_num_landmarks = 18
            assert self._hg_num_landmarks == 18

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
            for i in range(self._hg_num_modules):
                with tf.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, h = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    self.summary.feature_maps('hmap%d' % i, h, data_format=self._data_format_longer)
                    if y1 is not None:
                        metrics['heatmap%d_mse' % (i + 1)] = _tf_mse(h, y1)
                    x_prev = x
            if y1 is not None:
                loss_terms['heatmaps_mse'] = tf.reduce_mean([
                    metrics['heatmap%d_mse' % (i + 1)] for i in range(self._hg_num_modules)
                ])
            x = h
            outputs['heatmaps'] = x

        # Soft-argmax
        x = self._calculate_landmarks(x)
        with tf.variable_scope('upscale'):
            # Upscale since heatmaps are half-scale of original image
            x *= self._hg_first_layer_stride
            if y2 is not None:
                metrics['landmarks_mse'] = _tf_mse(x, y2)
            outputs['landmarks'] = x

        # Fully-connected layers for radius regression
        with tf.variable_scope('radius'):
            x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
            for i in range(3):
                with tf.variable_scope('fc%d' % (i + 1)):
                    x = tf.nn.relu(self._apply_bn(self._apply_fc(x, 100)))
            with tf.variable_scope('out'):
                x = self._apply_fc(x, 1)
            outputs['radius'] = x
            if y3 is not None:
                metrics['radius_mse'] = _tf_mse(tf.reshape(x, [-1]), y3)
                loss_terms['radius_mse'] = 1e-7 * metrics['radius_mse']
            self.summary.histogram('radius', x)

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

            with tf.variable_scope('hmap'):
                h = self._apply_conv(x_now, self._hg_num_landmarks, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.variable_scope('merge'):
                with tf.variable_scope('h'):
                    x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_hmaps
        return x_next, h

    _softargmax_coords = None

    def _calculate_landmarks(self, x):
        """Estimate landmark location from heatmaps."""
        with tf.variable_scope('argsoftmax'):
            if self._data_format == 'NHWC':
                _, h, w, _ = x.shape.as_list()
            else:
                _, _, h, w = x.shape.as_list()
            if self._softargmax_coords is None:
                # Assume normalized coordinate [0, 1] for numeric stability
                ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                             np.linspace(0, 1.0, num=h, endpoint=True),
                                             indexing='xy')
                ref_xs = np.reshape(ref_xs, [-1, h*w])
                ref_ys = np.reshape(ref_ys, [-1, h*w])
                self._softargmax_coords = (
                    tf.constant(ref_xs, dtype=tf.float32),
                    tf.constant(ref_ys, dtype=tf.float32),
                )
            ref_xs, ref_ys = self._softargmax_coords

            # Assuming N x 18 x 45 x 75 (NCHW)
            beta = 1e2
            if self._data_format == 'NHWC':
                x = tf.transpose(x, (0, 3, 1, 2))
            x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
            x = tf.nn.softmax(beta * x, axis=-1)
            lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[2])
            lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[2])

            # Return to actual coordinates ranges
            return tf.stack([
                lmrk_xs * (w - 1.0) + 0.5,
                lmrk_ys * (h - 1.0) + 0.5,
            ], axis=2)  # N x 18 x 2


def estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                 initial_gaze=None):
    """Given iris edge landmarks and other coordinates, estimate gaze direction.

    More correctly stated, estimate gaze from iris edge landmark coordinates, iris centre
    coordinates, eyeball centre coordinates, and eyeball radius in pixels.
    """
    e_x0, e_y0 = eyeball_centre
    i_x0, i_y0 = iris_centre

    if initial_gaze is not None:
        theta, phi = initial_gaze
        # theta = -theta
    else:
        theta = np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

    delta = 0.1 * np.pi
    if iris_landmarks[0, 0] < iris_landmarks[4, 0]:  # flipped
        alphas = np.flip(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0), axis=0)
    else:
        alphas = np.arange(-np.pi, np.pi, step=np.pi/4.0) + np.pi/4.0
    sin_alphas = np.sin(alphas)
    cos_alphas = np.cos(alphas)

    def gaze_fit_loss_func(inputs):
        theta, phi, delta, phase = inputs
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        # sin_alphas_shifted = np.sin(alphas + phase)
        sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = np.cos(alphas + phase)
        cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        # x = -np.cos(theta + delta * sin_alphas_shifted)
        x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x *= np.sin(phi + delta * cos_alphas_shifted)
        x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        x = x1 * x2
        # y = np.sin(theta + delta * sin_alphas_shifted)
        y1 = sin_theta * cos_delta_sin
        y2 = cos_theta * sin_delta_sin
        y = y1 + y2

        ix = e_x0 + eyeball_radius * x
        iy = e_y0 + eyeball_radius * y
        dx = ix - iris_landmarks[:, 0]
        dy = iy - iris_landmarks[:, 1]
        out = np.mean(dx ** 2 + dy ** 2)

        # In addition, match estimated and actual iris centre
        iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        out += iris_dx ** 2 + iris_dy ** 2

        # sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase
        dsin_alphas_shifted_dphase = -sin_alphas * sin_phase + cos_alphas * cos_phase
        dcos_alphas_shifted_dphase = -cos_alphas * sin_phase - sin_alphas * cos_phase

        # sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        # sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        # cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        # cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        dsin_delta_sin_ddelta = cos_delta_sin * sin_alphas_shifted
        dsin_delta_cos_ddelta = cos_delta_cos * cos_alphas_shifted
        dcos_delta_sin_ddelta = -sin_delta_sin * sin_alphas_shifted
        dcos_delta_cos_ddelta = -sin_delta_cos * cos_alphas_shifted
        dsin_delta_sin_dphase = cos_delta_sin * delta * dsin_alphas_shifted_dphase
        dsin_delta_cos_dphase = cos_delta_cos * delta * dcos_alphas_shifted_dphase
        dcos_delta_sin_dphase = -sin_delta_sin * delta * dsin_alphas_shifted_dphase
        dcos_delta_cos_dphase = -sin_delta_cos * delta * dcos_alphas_shifted_dphase

        # x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        dx1_dtheta = sin_theta * cos_delta_sin + cos_theta * sin_delta_sin
        dx2_dtheta = 0.0
        dx1_dphi = 0.0
        dx2_dphi = cos_phi * cos_delta_cos - sin_phi * sin_delta_cos
        dx1_ddelta = -cos_theta * dcos_delta_sin_ddelta + sin_theta * dsin_delta_sin_ddelta
        dx2_ddelta = sin_phi * dcos_delta_cos_ddelta + cos_phi * dsin_delta_cos_ddelta
        dx1_dphase = -cos_theta * dcos_delta_sin_dphase + sin_theta * dsin_delta_sin_dphase
        dx2_dphase = sin_phi * dcos_delta_cos_dphase + cos_phi * dsin_delta_cos_dphase

        # y1 = sin_theta * cos_delta_sin
        # y2 = cos_theta * sin_delta_sin
        dy1_dtheta = cos_theta * cos_delta_sin
        dy2_dtheta = -sin_theta * sin_delta_sin
        dy1_dphi = 0.0
        dy2_dphi = 0.0
        dy1_ddelta = sin_theta * dcos_delta_sin_ddelta
        dy2_ddelta = cos_theta * dsin_delta_sin_ddelta
        dy1_dphase = sin_theta * dcos_delta_sin_dphase
        dy2_dphase = cos_theta * dsin_delta_sin_dphase

        # x = x1 * x2
        # y = y1 + y2
        dx_dtheta = dx1_dtheta * x2 + x1 * dx2_dtheta
        dx_dphi = dx1_dphi * x2 + x1 * dx2_dphi
        dx_ddelta = dx1_ddelta * x2 + x1 * dx2_ddelta
        dx_dphase = dx1_dphase * x2 + x1 * dx2_dphase
        dy_dtheta = dy1_dtheta + dy2_dtheta
        dy_dphi = dy1_dphi + dy2_dphi
        dy_ddelta = dy1_ddelta + dy2_ddelta
        dy_dphase = dy1_dphase + dy2_dphase

        # ix = w_2 + eyeball_radius * x
        # iy = h_2 + eyeball_radius * y
        dix_dtheta = eyeball_radius * dx_dtheta
        dix_dphi = eyeball_radius * dx_dphi
        dix_ddelta = eyeball_radius * dx_ddelta
        dix_dphase = eyeball_radius * dx_dphase
        diy_dtheta = eyeball_radius * dy_dtheta
        diy_dphi = eyeball_radius * dy_dphi
        diy_ddelta = eyeball_radius * dy_ddelta
        diy_dphase = eyeball_radius * dy_dphase

        # dx = ix - iris_landmarks[:, 0]
        # dy = iy - iris_landmarks[:, 1]
        ddx_dtheta = dix_dtheta
        ddx_dphi = dix_dphi
        ddx_ddelta = dix_ddelta
        ddx_dphase = dix_dphase
        ddy_dtheta = diy_dtheta
        ddy_dphi = diy_dphi
        ddy_ddelta = diy_ddelta
        ddy_dphase = diy_dphase

        # out = dx ** 2 + dy ** 2
        dout_dtheta = np.mean(2 * (dx * ddx_dtheta + dy * ddy_dtheta))
        dout_dphi = np.mean(2 * (dx * ddx_dphi + dy * ddy_dphi))
        dout_ddelta = np.mean(2 * (dx * ddx_ddelta + dy * ddy_ddelta))
        dout_dphase = np.mean(2 * (dx * ddx_dphase + dy * ddy_dphase))

        # iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        # iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        # out += iris_dx ** 2 + iris_dy ** 2
        dout_dtheta += 2 * eyeball_radius * (sin_theta * sin_phi * iris_dx + cos_theta * iris_dy)
        dout_dphi += 2 * eyeball_radius * (-cos_theta * cos_phi * iris_dx)

        return out, np.array([dout_dtheta, dout_dphi, dout_ddelta, dout_dphase])

    phase = 0.02
    result = scipy.optimize.minimize(gaze_fit_loss_func, x0=[theta, phi, delta, phase],
                                     bounds=(
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (0.01*np.pi, 0.5*np.pi),
                                         (-np.pi, np.pi),
                                     ),
                                     jac=True,
                                     tol=1e-6,
                                     method='TNC',
                                     options={
                                         # 'disp': True,
                                         'gtol': 1e-6,
                                         'maxiter': 100,
                                    })
    if result.success:
        theta, phi, delta, phase = result.x

    return np.array([-theta, phi])
