''' geoconv
Original author: Shiyi Lan
All Rights Reserved. 2018

'''
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
geoconv_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_geoconv_so.so'))

def aggregate(feat, xyz, radius, decay_radius, delta=0):
    '''
    inputs:
        feature: batch_size * num_points * num_channels     float32
        xyz: batch_size * num_points * 3                    float32
        radius:                                             float32
        decay_radius:                                       float32
        delta                                               int
    returns:
        output feature: batch_size * num_points * num_channels  float32
        norm feature: batch_size * num_points
    '''
    return geoconv_module.aggregate(feat, xyz, radius, decay_radius, delta)


@tf.RegisterGradient('Aggregate')
def _aggregate_grad(op, *out_g):
    feat = op.inputs[0]
    xyz  = op.inputs[1]
    top_g = out_g[0]
    norm_buffer = out_g[1]
    radius = op.get_attr("radius")
    decay_radius = op.get_attr("decayradius")
    delta = op.get_attr("delta")
    return [geoconv_module.aggregate_grad(feat, xyz, top_g, radius, decay_radius, delta)[0], None]


class AggregateTest(tf.test.TestCase):
    def test(self):
        pass

    def test_grad(self):
        with tf.device('/gpu:0'):
            feats = tf.constant(np.random.random((8, 128, 192)).astype('float32'))
            xyz   = tf.constant(np.random.random((8, 128, 3)).astype('float32'))
            ag, _ = aggregate(feats, xyz, 0.3, 0.6)
            print(ag)

        with self.test_session():
            print("------ Going to compute gradient error")
            err = tf.test.compute_gradient_error(feats, (8, 128, 192), ag, (8, 128, 32))
            print(err)
            self.assertLess(err, 1e-4)

if __name__ == "__main__":
    tf.test.main()
