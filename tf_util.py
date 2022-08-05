# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch
import numpy as np
import math

def mlp(features, layer_dims, bn=None, bn_params=None, prefix=''):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope=prefix + 'fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope=prefix + 'fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None, prefix=''):
    # inputs -> 1 x total n x 3
    # layers -> 16, 32 chanels
    # first weights - > 3 x 16 x 1 (in channels x out channels x kernel size)
    all_features = []
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope=prefix + 'conv_%d' % i)
        all_features.append(inputs)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope=prefix + 'conv_%d' % (len(layer_dims) - 1))
    all_features.append(outputs)
    return outputs, all_features


def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)

def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update

def add_test_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['test_summary'])
    return update

def earth_mover_shapenet(pcd1, pcd2, prob):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    cost = cost * prob
    return tf.reduce_mean(cost / num_points)

def chamfer_shapenet(pcd1, pcd2, prob):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    dist1 = dist1 * prob
    dist2 = dist2 * prob

    dist1 = tf.reduce_mean(dist1)
    dist2 = tf.reduce_mean(dist2)

    return (dist1 + dist2) / 2

def earth_mover_prob(pcd1, pcd2, prob, dropped_octants, undropped_octants):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    cost = cost * prob

    return tf.reduce_mean(cost / num_points), \
           tf.reduce_mean((cost * dropped_octants) / num_points), \
           tf.reduce_mean((cost * undropped_octants) / num_points)

def chamfer_prob(pcd1, pcd2, prob, loss_weight_gt_pred, dropped_octants, undropped_octants):
    # (B, N)
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    # square root, mean -> (B)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    # (B)
    dist1 = dist1 * prob
    dist2 = dist2 * prob

    dist1_dropped = dist1 * dropped_octants
    dist1_undropped = dist1 * undropped_octants
    dist2_dropped = dist2 * dropped_octants
    dist2_undropped = dist2 * undropped_octants

    # (1)
    dist1_m = tf.reduce_mean(dist1)
    dist2_m = tf.reduce_mean(dist2)

    dist1_m_dropped = tf.reduce_mean(dist1_dropped)
    dist1_m_undropped = tf.reduce_mean(dist1_undropped)
    dist2_m_dropped = tf.reduce_mean(dist2_dropped)
    dist2_m_undropped = tf.reduce_mean(dist2_undropped)

    chamfer_all = ((1 - loss_weight_gt_pred) * dist1_m) + (loss_weight_gt_pred * dist2_m)
    chamfer_dropped = ((1 - loss_weight_gt_pred) * dist1_m_dropped) + (loss_weight_gt_pred * dist2_m_dropped)
    chamfer_undropped = ((1 - loss_weight_gt_pred) * dist1_m_undropped) + (loss_weight_gt_pred * dist2_m_undropped)

    return chamfer_all, dist1, dist2, chamfer_dropped, chamfer_undropped
    # return ((1 - loss_weight_gt_pred) * dist1_m) + (loss_weight_gt_pred * dist2_m), \
    #        dist1, dist2,
    # return (dist1 + dist2) / 2

def repulsion_loss(pcd1, pcd2, prob, h_param=0.03):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    # no sqrt for dist1, because l2 norm is taken
    rep = (-dist1)*tf.math.exp(-(dist1*dist1)/(h_param*h_param))
    rep = tf.reduce_mean(rep, axis=1)
    rep = rep * prob
    rep_loss = tf.reduce_mean(rep)

    # dist1 = tf.reduce_mean(tf.sqrt(dist1))
    # dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return rep_loss

def chamfer_metric(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2

def chamfer_metric_prob(pcd1, pcd2, prob):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    dist1 = dist1 * prob
    dist2 = dist2 * prob

    dist1 = tf.reduce_mean(dist1)
    dist2 = tf.reduce_mean(dist2)

    return (dist1 + dist2) / 2

def chamfer_one_direction(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    # dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return dist1


def chamfer_shapenet_one_direction(pcd1, pcd2, prob):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    # dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    dist1 = dist1 * prob
    # dist2 = dist2 * prob

    dist1 = tf.reduce_mean(dist1)
    # dist2 = tf.reduce_mean(dist2)

    return dist1

def chamfer_weighted(pcd1, pcd2, gt_to_pred_weight, pred_to_gt_weight):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2


def chamfer_shapenet_weighted(pcd1, pcd2, prob, gt_to_pred_weight, pred_to_gt_weight):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    dist1 = dist1 * prob
    dist2 = dist2 * prob

    dist1 = tf.reduce_mean(dist1)
    dist2 = tf.reduce_mean(dist2)

    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2

def chamfer_one_direction_minofn(pcd1, pcd2, batch_size, num_k = 10):

    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1_mean = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist1_re = tf.reshape(dist1_mean, [-1, batch_size])

    dist1_topk_val, dist1_topk_idx = tf.nn.top_k(dist1_re, k = num_k)

    dist1_mean = tf.reduce_mean(dist1_topk_val)
    # dist2 = tf.reduce_mean(tf.sqrt(dist2))

    return dist1_mean

def chamfer_shapenet_one_direction_minofn(pcd1, pcd2, prob, batch_size, num_k = 10):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    # dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    dist1_p = dist1 * prob
    # dist2 = dist2 * prob
    dist1_re = tf.reshape(dist1_p, [-1, batch_size])

    dist1_topk_val, dist1_topk_idx = tf.nn.top_k(dist1_re, k = num_k)

    dist1 = tf.reduce_mean(dist1_topk_val)
    # dist2 = tf.reduce_mean(dist2)

    return dist1, dist1_topk_val, dist1_p

def chamfer_weighted_hindsight(pcd1, pcd2, gt_to_pred_weight,
                               pred_to_gt_weight, num_poses=4):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2) # B x N -> num_poses x actual_b x N

    dist1 = tf.reshape(dist1, [num_poses, -1])
    dist2 = tf.reshape(dist2, [num_poses, -1])
    dist1 = tf.reduce_min(dist1, axis=0)
    dist2 = tf.reduce_min(dist2, axis=0)

    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2

def chamfer_weighted_hindsight_updated(pcd1, pcd2, gt_to_pred_weight,
                               pred_to_gt_weight, num_poses=4):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2) # B x N -> num_poses x actual_b x N

    dist1 = tf.reshape(dist1, [num_poses, 32, -1]) # 4 x B x N
    dist2 = tf.reshape(dist2, [num_poses, 32, -1])

    dist1_all = tf.reduce_mean(tf.sqrt(dist1), [1, 2]) # 4
    dist2_all = tf.reduce_mean(tf.sqrt(dist2), [1, 2]) # 4

    dist1 = tf.reduce_min(dist1_all, axis=0)
    dist2 = tf.reduce_min(dist2_all, axis=0)

    # dist1 = tf.reduce_mean(tf.sqrt(dist1))
    # dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2, dist1_all, dist2_all


def chamfer_weighted_hindsight_updated_object(pcd1, pcd2, gt_to_pred_weight,
                               pred_to_gt_weight, num_poses=4):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2) # B x N -> num_poses x actual_b x N

    dist1 = tf.reshape(dist1, [num_poses, 8, -1]) # 4 x B x 4.N
    dist2 = tf.reshape(dist2, [num_poses, 8, -1]) # 4 x B x 4.N

    dist1_all = tf.reduce_mean(tf.sqrt(dist1), axis=2) # 4 x B
    dist2_all = tf.reduce_mean(tf.sqrt(dist2), axis=2) # 4 x B

    dist1 = tf.reduce_min(dist1_all, axis=0) # B
    dist2 = tf.reduce_min(dist2_all, axis=0) # B

    dist1 = tf.reduce_mean(tf.sqrt(dist1)) # 1
    dist2 = tf.reduce_mean(tf.sqrt(dist2)) # 1
    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2, dist1_all, dist2_all

def chamfer_weighted_hindsight_updated_view(pcd1, pcd2, gt_to_pred_weight,
                               pred_to_gt_weight, num_poses=4):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2) # B x N -> num_poses x actual_b x N

    dist1 = tf.reshape(dist1, [num_poses, 32, -1]) # 4 x B.4 x N
    dist2 = tf.reshape(dist2, [num_poses, 32, -1]) # 4 x B.4 x N

    dist1_all = tf.reduce_mean(tf.sqrt(dist1), axis=2) # 4 x B.4
    dist2_all = tf.reduce_mean(tf.sqrt(dist2), axis=2) # 4 x B.4

    dist1 = tf.reduce_min(dist1_all, axis=0) # B.4
    dist2 = tf.reduce_min(dist2_all, axis=0) # B.4

    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2, dist1_all, dist2_all

def chamfer_shapenet_weighted_hindsight(pcd1, pcd2, prob, gt_to_pred_weight, pred_to_gt_weight, num_poses=4):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1), axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2), axis=1)

    # # B

    dist1_val = dist1 * prob + ((1 - prob) * tf.constant(1e10, dtype=tf.float32, shape=prob.shape))# [2.3, 4.5, 2] * [0, 1, 0] = [0, 4.5, 0] + [inf, 0, inf] = [inf, 4.5, inf]
    dist2_val = dist2 * prob + ((1 - prob) * tf.constant(1e10, dtype=tf.float32, shape=prob.shape))

    dist1_val_re = tf.reshape(dist1_val, [num_poses, -1])
    dist2_val_re = tf.reshape(dist2_val, [num_poses, -1])
    prob_re = tf.reshape(prob, [num_poses, -1])[0] # prob -> 128
    dist1_min = tf.reduce_min(dist1_val_re, axis=0) * prob_re
    dist2_min = tf.reduce_min(dist2_val_re, axis=0) * prob_re

    dist1 = tf.reduce_mean(dist1_min)
    dist2 = tf.reduce_mean(dist2_min)

    return gt_to_pred_weight * dist1 + pred_to_gt_weight * dist2, [dist1, dist2, \
            dist1_val, dist2_val, dist1_val_re, dist2_val_re, dist1_min, dist2_min]


def gen_grid_up(up_ratio):
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_ratio%i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.linspace(-0.2, 0.2, num_x)
    grid_y = tf.linspace(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if weight_decay>0:
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer=None
        outputs = tf.contrib.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                            activation_fn=activation_fn,weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=regularizer,
                                           biases_regularizer=regularizer)
        return outputs