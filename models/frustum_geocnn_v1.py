''' Frsutum PointNets v1 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import geoconv_util3
from tf_util    import perceptron
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss

def get_instance_seg_v1_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value


    xyz = point_cloud[:, :, :3]
    net = geoconv_util3.geoconv(point_cloud, xyz, 64, 32, 0.05, 0.15,
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    vis = net
    net = geoconv_util3.geoconv(net, xyz, 64, 32, 0.1, 0.2,
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = geoconv_util3.geoconv(net, xyz, 64, 32, 0.15, 0.3,
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    point_feat = net

    net = geoconv_util3.geoconv(net, xyz, 128, 64, 0.2, 0.4,
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = geoconv_util3.geoconv(net, xyz, 1024, 128, 0.2, 0.4,
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    net = tf.expand_dims(net, 2)

    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    global_feat = tf.concat([tf.squeeze(global_feat, 1), tf.expand_dims(one_hot_vec, 1)], axis=2)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1])
    concat_feat = tf.concat(axis=2, values=[point_feat, global_feat_expand])

    net = geoconv_util3.geoconv(concat_feat, xyz, 512, 32, 0.15, 0.3,
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = geoconv_util3.geoconv(net, xyz, 256, 32, 0.1, 0.2,
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = geoconv_util3.geoconv(net, xyz, 128, 32, 0.05, 0.1,
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = geoconv_util3.geoconv(net, xyz, 128, 32, 0.05, 0.1,
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    end_points['vis'] = vis

    logits = tf_util.perceptron(net, 2, 
                         activation_fn=None,
                         scope='conv10')
    return logits, end_points
 

def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    ''' 
    num_point = object_point_cloud.get_shape()[1].value
    net = tf_util.perceptron(object_point_cloud, 128,
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.perceptron(net, 128,
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.perceptron(net, 256,
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.perceptron(net, 512,
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(tf.expand_dims(net, 3),  [num_point,1],
                         padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1, 3])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
        is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}
    
    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v1_net(\
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(\
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v1_net(\
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        outputs = get_model(inputs, tf.ones((32,3)), tf.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.zeros((32,1024),dtype=tf.int32),
            tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,3)), outputs)
        print(loss)
