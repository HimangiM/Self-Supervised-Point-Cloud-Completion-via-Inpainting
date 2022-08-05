import argparse
import importlib
import os
import random
import sys

import numpy as np
import tensorflow as tf
from termcolor import colored

from data_util_octants_semantickitti_train import \
    lmdb_dataflow
from data_util_octants_semantickitti_test import \
    lmdb_dataflow_test
from tf_util import add_train_summary


sys.path.append('/home/hmittal/point_completion_net/differentiable-point-clouds/dpc')
sys.path.append('/home/hmittal/differentiable-point-clouds/dpc')
# sys.path.append('/home/hmittal/differentiable-point-clouds/dpc')
print (sys)
from util.point_cloud import point_cloud_distance
# from util.app_config import config as app_config
from util.tools import partition_range
import pandas as pd

def log_string(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)

def resample_pcd(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
    return pc[idx[:n]]

def return_octants(points, pts_threshold=128, fuzzy_threshold=0):
    points_mean = np.mean(bbox, 0)
    x_center, y_center, z_center = points_mean[0], points_mean[1], points_mean[2]

    data1_points = np.logical_and(np.logical_and(points[:, 0] < (x_center + fuzzy_threshold),
                                                 points[:, 1] < (y_center + fuzzy_threshold)),
                                                 points[:, 2] < (z_center + fuzzy_threshold))
    if np.sum(data1_points) < pts_threshold:
        data1 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data1 = points[data1_points, :] # if points in an octant are less than pts_threshold,
                                        # then octant has pts_threshold num. of zeros

    data2_points = np.logical_and(np.logical_and(points[:, 0] < (x_center + fuzzy_threshold),
                                                 points[:, 1] < (y_center + fuzzy_threshold)),
                                                 points[:, 2] > (z_center - fuzzy_threshold))
    if np.sum(data2_points) < pts_threshold:
        data2 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data2 = points[data2_points, :]

    data3_points = np.logical_and(np.logical_and(points[:, 0] < (x_center + fuzzy_threshold),
                                                 points[:, 1] > (y_center - fuzzy_threshold)),
                                                 points[:, 2] < (z_center + fuzzy_threshold))
    if np.sum(data3_points) < pts_threshold:
        data3 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data3 = points[data3_points, :]

    data4_points = np.logical_and(np.logical_and(points[:, 0] < (x_center + fuzzy_threshold),
                                                 points[:, 1] > (y_center - fuzzy_threshold)),
                                                 points[:, 2] > (z_center - fuzzy_threshold))
    if np.sum(data4_points) < pts_threshold:
        data4 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data4 = points[data4_points, :]

    data5_points = np.logical_and(np.logical_and(points[:, 0] > (x_center - fuzzy_threshold),
                                                 points[:, 1] < (y_center + fuzzy_threshold)),
                                                 points[:, 2] < (z_center + fuzzy_threshold))
    if np.sum(data5_points) < pts_threshold:
        data5 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data5 = points[data5_points, :]

    data6_points = np.logical_and(np.logical_and(points[:, 0] > (x_center - fuzzy_threshold),
                                                 points[:, 1] < (y_center + fuzzy_threshold)),
                                                 points[:, 2] > (z_center - fuzzy_threshold))
    if np.sum(data6_points) < pts_threshold:
        data6 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data6 = points[data6_points, :]

    data7_points = np.logical_and(np.logical_and(points[:, 0] > (x_center - fuzzy_threshold),
                                                 points[:, 1] > (y_center - fuzzy_threshold)),
                                                 points[:, 2] < (z_center + fuzzy_threshold))
    if np.sum(data7_points) < pts_threshold:
        data7 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data7 = points[data7_points, :]

    data8_points = np.logical_and(np.logical_and(points[:, 0] > (x_center - fuzzy_threshold),
                                                 points[:, 1] > (y_center - fuzzy_threshold)),
                                                 points[:, 2] > (z_center - fuzzy_threshold))
    if np.sum(data8_points) < pts_threshold:
        data8 = np.zeros((pts_threshold, 3)).astype(np.float32)
    else:
        data8 = points[data8_points, :]

    return [data1, data2, data3, data4, data5, data6, data7, data8]

def compute_distance(cfg, sess, min_dist, idx, source, target, source_np, target_np):
    """
    compute projection from source to target
    """
    num_parts = cfg.pc_eval_chamfer_num_parts
    partition = partition_range(source_np.shape[0], num_parts)
    min_dist_np = np.zeros((0,))
    idx_np = np.zeros((0,))
    for k in range(num_parts):
        r = partition[k, :]
        src = source_np[r[0]:r[1]]
        (min_dist_0_np, idx_0_np) = sess.run([min_dist, idx],
                                             feed_dict={source: src,
                                                       target: target_np})
        min_dist_np = np.concatenate((min_dist_np, min_dist_0_np), axis=0)
        idx_np = np.concatenate((idx_np, idx_0_np), axis=0)
    return min_dist_np, idx_np

def test(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_octants_pl = [tf.placeholder(tf.float32, (1, None, 3), 'inputs') for _ in range(args.num_parts)]
    npts_octants_pl = [tf.placeholder(tf.int32, (args.batch_size,), 'num_points') for _ in range(args.num_parts)]
    gt_octants_pl = [tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points_octants, 3), 'ground_truths')
             for _ in range(args.num_parts)]
    prob_octants_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))
    missing_octants_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))

    inputs_pcn_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    npts_pcn_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points')
    gt_pcn_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points_pcn, 3), 'ground_truths')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_octants_pl, npts_octants_pl, gt_octants_pl, alpha,
                               args.num_parts, prob_octants_pl,
                               args.encoder_dropout, missing_octants_pl, inputs_pcn_pl,
                               npts_pcn_pl, gt_pcn_pl, args.gt_to_pred_weight, args.pred_to_gt_weight)
    # model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha, args.num_parts, prob_pl,
    #                            args.encoder_dropout, missing_octants_pl)

    source_pc = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    target_pc = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    _, min_dist, min_idx = point_cloud_distance(source_pc, target_pc)

    df_test, num_test = lmdb_dataflow_test(args.lmdb_test, args.batch_size,
                                        args.num_input_points_octants, args.num_gt_points_octants,
                                        args.num_input_points_pcn, args.num_gt_points_pcn,
                                        args.fuzzy_boundary, is_training=False)
    test_gen = df_test.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=50)

    saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))

    num_eval_steps = num_test // args.batch_size

    chamfer_l = []
    precision_l = []
    coverage_l = []
    chamfer_dict = dict()
    chamfer_dict['id'] = []
    chamfer_dict['chamfer_val'] = []
    chamfer_dict['precision_val'] = []
    chamfer_dict['coverage_val'] = []

    for e in range(num_eval_steps):

        ids, inputs_octants, npts_octs, gts_octants, missing_octants, \
        inputs_pcn, npts_pcn, gts_pcn, gt_original_pcn = next(test_gen)

        # missing_octants: B x 8: what all octants of each pc in B are all zero
        for i in range(args.batch_size):
            if np.sum(gts_octants[:, i, :, :]**2) == 0:
        # if gt.shape[1] < args.batch_size:
                print('batch incomplete. skipping', gts_octants.shape, args.batch_size)
                continue

        npts_octant = np.sum(npts_octs, axis=1) # (8,)
        inputs_org_octants = np.split(inputs_octants[0], np.cumsum(npts_octant)[:-1])

        sep_inputs = []
        for i in range(8):
            sep_inputs.append(np.split(inputs_org_octants[i], np.cumsum(npts_octs[i])[:-1]))

        feed_dict = {}
        for i in range(8):
            # feed_dict[inputs_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
            feed_dict[inputs_octants_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
            feed_dict[npts_octants_pl[i]] = npts_octs[i]
            feed_dict[gt_octants_pl[i]] = gts_octants[i]
            feed_dict[missing_octants_pl] = missing_octants
            # print ('Shape here:', i, inputs_org_octants[i].shape)

        feed_dict[inputs_pcn_pl] = inputs_pcn
        feed_dict[npts_pcn_pl] = npts_pcn
        feed_dict[gt_pcn_pl] = gts_pcn
        feed_dict[prob_octants_pl] = missing_octants
        feed_dict[is_training_pl] = False

        fine_output = sess.run(model.outputs, feed_dict=feed_dict)
        fine_output = np.array(fine_output)

        res = np.concatenate(fine_output[0], axis=1)
        output_concat = np.concatenate((res, fine_output[1]), axis=1)
        # output_concat = np.concatenate(fine_output, axis=1)

        # if e < 10:
        #     np.savez(os.path.join(args.results_dir, 'test_output_{}'.format(e)),
        #              output = output_concat, gt = gt_original_pcn,
        #              inputs = sep_inputs, ids = ids)

        # import ipdb; ipdb.set_trace()
        inputs_ = np.split(inputs_pcn[0], np.cumsum(npts_pcn)[:-1])

        e_l = [104, 206, 133, 318, 400, 401]
        if e in e_l:
            np.savez(os.path.join(args.results_dir, 'test_output_{}'.format(e)),
                     output = output_concat, gt = gt_original_pcn,
                     inputs = inputs_, ids = ids)

        for bv in range(args.batch_size):
            ids_bv = ids[bv]
            input_points = inputs_[bv]
            gt_points = gt_original_pcn[bv].reshape(-1, 3)
            output_points = output_concat[bv].reshape(-1, 3)

            (min_dist_0_np, idx_0_np) = sess.run([min_dist, min_idx],
                                             feed_dict={source_pc: gt_points,
                                                       target_pc: output_points})

            (min_dist_1_np, idx_1_np) = sess.run([min_dist, min_idx],
                                                     feed_dict={source_pc: output_points,
                                                               target_pc: gt_points})

            chamfer_val = np.sum([np.mean(min_dist_1_np), np.mean(min_dist_0_np)])
            precision_val = np.mean(min_dist_1_np)
            coverage_val = np.mean(min_dist_0_np)
            chamfer_l.append(chamfer_val)
            precision_l.append(precision_val)
            coverage_l.append(coverage_val)

            # if chamfer_val <= 0.06:
            #     np.savez(os.path.join(args.results_dir, 'test_output_{}_{}'.format(e, bv)),
            #          output = output_points, gt = gt_points,
            #          inputs = input_points, ids = ids_bv)

            print ('Step, chamfer, precision, coverage:', e, np.mean(chamfer_l), np.mean(precision_l), np.mean(coverage_l))
            chamfer_dict['id'].append(ids[bv])
            chamfer_dict['chamfer_val'].append(chamfer_val)
            chamfer_dict['precision_val'].append(precision_val)
            chamfer_dict['coverage_val'].append(coverage_val)

            # np.savez(os.path.join(args.results_dir, 'test_output_{}_{}'.format(e, bv)),
            #          output = output_points, gt = gt_points,
            #          inputs = sep_inputs)

    print ('Final mean, chamfer, precision, recall:', np.mean(chamfer_l), np.mean(precision_l), np.mean(coverage_l))
    print ('Final st dev, chamfer, precision, recall:', np.std(chamfer_l), np.std(precision_l), np.std(coverage_l))

    df = pd.DataFrame(chamfer_dict)
    df.to_csv(os.path.join(args.results_dir, 'results.csv'))


def valid(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_pl = [tf.placeholder(tf.float32, (1, None, 3), 'inputs') for _ in range(args.num_parts)]
    npts_pl = [tf.placeholder(tf.int32, (args.batch_size,), 'num_points') for _ in range(args.num_parts)]
    gt_pl = [tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')
             for _ in range(args.num_parts)]
    prob_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))
    missing_octants_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha, args.num_parts, prob_pl,
                               args.encoder_dropout, missing_octants_pl)

    df_valid, num_valid = lmdb_dataflow_valid(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen= df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))

    num_eval_steps = num_valid // args.batch_size

    for i in range(num_eval_steps):
        prob = np.random.choice(2, p=[0.2,0.8], size=(args.batch_size, args.num_parts)).astype(np.float32)

        ids, inputs_org, npts, gt, gt_original, missing_octants, _ = next(valid_gen)
        npts_octant = np.sum(npts, axis=1) # (8,)  # (1500*batch, 1500*batch......1500*batch)
        inputs_org_octants = np.split(inputs_org[0], np.cumsum(npts_octant)[:-1]) # [(N_1 + N_2 + ... + N_8) x 3] -> [(N_1 x 3), (N_2 x 3), ..., (N_8 x 3)]

        sep_inputs = [] # (8, 4)
        for j in range(8):
            sep_inputs.append(np.split(inputs_org_octants[j], np.cumsum(npts[j])[:-1]))

        # import ipdb; ipdb.set_trace()
        # inputs_org_l = []   # list of 8
        # pt = prob.T         # (8, 4)
        # for x in range(8):
        #     i_l = []        # new list for every octant
        #     n_sum = np.concatenate(([0], np.cumsum(npts[x])))
        #     for y in range(args.batch_size):
        #         if pt[x][y] == 0:
        #             i_l.append(np.zeros((n_sum[y+1] - n_sum[y], 3)))
        #         else:
        #             i_l.append(inputs_org_octants[x][n_sum[y]:n_sum[y+1]])
        #
        #     i_l = np.concatenate((np.array(i_l)))
        #     inputs_org_l.append(i_l)

        # # one eighth - randomly select one octant for each pc in batch size to remove
        # octants_to_remove = np.random.randint(8, size=(args.batch_size, 1)) # [0, 7, 3, 5]
        # # one fourth - randomly select two octants for each pc in batch size to remove
        # # octants_to_remove = np.random.randint(8, size=(args.batch_size, 2)) # [[0, 4], [3, 6], [8, 1], [4, 3]
        # # half - randomly select four octants for each pc in batch size to remove
        # # octants_to_remove = np.random.randint(8, size=(args.batch_size, 4)) # [[0, 5, 2, 4], ...]
        #
        # octant_removal_masks = []
        # for k in range(8):
        #     octant_removal_mask = []
        #     for b in range(args.batch_size):
        #         if k in octants_to_remove[b]:
        #             octant_removal_mask.append(np.zeros((npts[k][b], 1)))  # 1500, 1
        #         else:
        #             octant_removal_mask.append(np.ones((npts[k][b], 1))) # 1500, 1
        #     octant_removal_masks.append(np.concatenate(octant_removal_mask)) # octant wise - ith row - 1500*4 - 6000
        #     # (6000 * 3) * 6000
        #     inputs_org_octants[k] = inputs_org_octants[k] * np.concatenate(octant_removal_mask).astype(np.float32) # N_k x 3 * N_k x 1
        #
        # for k in range(8):
        #     for b in range(args.batch_size):
        #         if k in octants_to_remove[b]:
        #             sep_inputs[k][b] = sep_inputs[k][b] * 0

        # import ipdb; ipdb.set_trace()
        feed_dict = {}
        for k in range(8):
            feed_dict[inputs_pl[k]] = np.expand_dims(inputs_org_octants[k], axis=0)
            feed_dict[npts_pl[k]] = npts[k]
            feed_dict[gt_pl[k]] = gt[k]
            feed_dict[missing_octants_pl] = missing_octants

        feed_dict[prob_pl] = prob
        feed_dict[is_training_pl] = False
        fine_output = sess.run(model.outputs, feed_dict=feed_dict)

        fine_output = np.array(fine_output)
        fine_output = fine_output.reshape(-1, 3)
        fine_output = np.expand_dims(fine_output, axis=0)

        np.savez(os.path.join(args.results_dir, 'test_output_{}'.format(i)),
                 partial_input1 = sep_inputs[0], partial_input2 = sep_inputs[1],
                 partial_input3 = sep_inputs[2], partial_input4 = sep_inputs[3],
                 partial_input5 = sep_inputs[4], partial_input6 = sep_inputs[5],
                 partial_input7 = sep_inputs[6], partial_input8 = sep_inputs[7],
                 complete_output = fine_output, ground_truth = gt_original,
                 inputs_lmdb = inputs_org, npts = npts, ids = ids, prob = prob)

# def valid(args):
#     global_step = tf.Variable(0, trainable=False, name='global_step')
#     alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
#                                         [0.01, 0.1, 0.5, 1.0], 'alpha_op')
#     inputs_pl = [tf.placeholder(tf.float32, (1, None, 3), 'inputs') for _ in range(args.num_parts)]
#     npts_pl = [tf.placeholder(tf.int32, (args.batch_size,), 'num_points') for _ in range(args.num_parts)]
#     gt_pl = [tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')
#              for _ in range(args.num_parts)]
#     prob_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))
#
#     model_module = importlib.import_module('.%s' % args.model_type, 'models')
#     model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha, args.num_parts, prob_pl,
#                                args.encoder_dropout)
#
#     df_valid, num_valid = lmdb_dataflow(
#         args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
#     valid_gen= df_valid.get_data()
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.allow_soft_placement = True
#     sess = tf.Session(config=config)
#     saver = tf.train.Saver()
#
#     saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
#
#     num_eval_steps = num_valid // args.batch_size
#
#     for i in range(num_eval_steps):
#
#         ids, inputs_org, npts, gt = next(valid_gen)
#         npts_octant = np.sum(npts, axis=1) # (8,)
#         inputs_org_octants = np.split(inputs_org[0], np.cumsum(npts_octant)[:-1])
#
#         sep_inputs = []
#         for j in range(8):
#             sep_inputs.append(np.split(inputs_org_octants[j], np.cumsum(npts[j])[:-1]))
#
#         feed_dict = {}
#         for k in range(8):
#             feed_dict[inputs_pl[k]] = np.expand_dims(inputs_org_octants[k], axis=0)
#             feed_dict[npts_pl[k]] = npts[k]
#             feed_dict[gt_pl[k]] = gt[k]
#
#         fine_output = sess.run([model.outputs], feed_dict=feed_dict)
#
#         np.savez(os.path.join(args.results_dir, 'test_output_{}'.format(i)),
#                  partial_input1 = sep_inputs[0], partial_input2 = sep_inputs[1],
#                  partial_input3 = sep_inputs[2], partial_input4 = sep_inputs[3],
#                  partial_input5 = sep_inputs[4], partial_input6 = sep_inputs[5],
#                  partial_input7 = sep_inputs[6], partial_input8 = sep_inputs[7],
#                  complete_output = fine_output, ground_truth = gt,
#                  inputs_lmdb = inputs_org, npts = npts, ids = ids)

def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_octants_pl = [tf.placeholder(tf.float32, (1, None, 3), 'inputs') for _ in range(args.num_parts)]
    npts_octants_pl = [tf.placeholder(tf.int32, (args.batch_size,), 'num_points') for _ in range(args.num_parts)]
    gt_octants_pl = [tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points_octants, 3), 'ground_truths')
             for _ in range(args.num_parts)]
    prob_octants_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))
    missing_octants_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_parts))

    inputs_pcn_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    npts_pcn_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points')
    gt_pcn_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points_pcn, 3), 'ground_truths')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_octants_pl, npts_octants_pl, gt_octants_pl, alpha,
                               args.num_parts, prob_octants_pl,
                               args.encoder_dropout, missing_octants_pl, inputs_pcn_pl,
                               npts_pcn_pl, gt_pcn_pl, args.gt_to_pred_weight, args.pred_to_gt_weight)
    # model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha, args.num_parts, prob_pl,
    #                            args.encoder_dropout, missing_octants_pl)
    add_train_summary('alpha', alpha)

    LOG_FOUT = open(args.slurm_output_file, 'w')

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')
    train_summary = tf.summary.merge_all('train_summary')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    # source_pc = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    # target_pc = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    #
    # _, min_dist, min_idx = point_cloud_distance(source_pc, target_pc)

    df_train, num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                        args.num_input_points_octants, args.num_gt_points_octants,
                                        args.num_input_points_pcn, args.num_gt_points_pcn,
                                        args.fuzzy_boundary, is_training=True)
    train_gen = df_train.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=50)

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_restore_dir))
        writer = tf.summary.FileWriter(args.log_dir)
        print ('Log dir restored')
    else:
        sess.run(tf.global_variables_initializer())
        os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))  # bkp of model def
        os.system('cp train.py %s' % args.log_dir)                         # bkp of train procedure
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    init_step = sess.run(global_step)

    # inputs_org = (1, N, 3), gt = (4, 2048, 3)
    prob_l = []

    for step in range(init_step+1, args.max_step+1):
        prob = np.random.choice(2, p=[0.5,0.5], size=(args.batch_size, args.num_parts)).astype(np.float32)

        for i in range(args.batch_size):
            idx = np.random.choice(args.num_parts, random.sample(range(args.prob_zero_start, args.prob_zero_end), 1), replace=False)
            prob[i][idx] = 0

        prob_l.append(prob)

        # ids, inputs_org, npts, gt, missing_octants, gt_original = next(train_gen) # inputs -> 1 x total_points x 3, npts -> 8 X B, gts -> 8 X B X gt X 3
        ids, inputs_octants, npts_octs, gts_octants, missing_octants, \
               inputs_pcn, npts_pcn, gts_pcn = next(train_gen)

        # missing_octants: B x 8: what all octants of each pc in B are all zero
        for i in range(args.batch_size):
            if np.sum(gts_octants[:, i, :, :]**2) == 0:
        # if gt.shape[1] < args.batch_size:
                print('batch incomplete. skipping', gts_octants.shape, args.batch_size)
                continue

        npts_octant = np.sum(npts_octs, axis=1) # (8,)
        inputs_org_octants = np.split(inputs_octants[0], np.cumsum(npts_octant)[:-1])

        sep_inputs = []
        for i in range(8):
            sep_inputs.append(np.split(inputs_org_octants[i], np.cumsum(npts_octs[i])[:-1]))

        # import ipdb; ipdb.set_trace()
        # inputs_org_l = []   # list of 8
        # pt = prob.T         # (8, 4)
        # for x in range(8):
        #     i_l = []        # new list for every octant
        #     n_sum = np.concatenate(([0], np.cumsum(npts[x])))
        #     for y in range(args.batch_size):
        #         if pt[x][y] == 0:
        #             i_l.append(np.zeros((n_sum[y+1] - n_sum[y], 3)))
        #         else:
        #             i_l.append(inputs_org_octants[x][n_sum[y]:n_sum[y+1]])
        #
        #     i_l = np.concatenate((np.array(i_l)))
        #     inputs_org_l.append(i_l)

        feed_dict = {}
        for i in range(8):
            # feed_dict[inputs_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
            feed_dict[inputs_octants_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
            feed_dict[npts_octants_pl[i]] = npts_octs[i]
            feed_dict[gt_octants_pl[i]] = gts_octants[i]
            feed_dict[missing_octants_pl] = missing_octants
            # print ('Shape here:', i, inputs_org_octants[i].shape)

        feed_dict[inputs_pcn_pl] = inputs_pcn
        feed_dict[npts_pcn_pl] = npts_pcn
        feed_dict[gt_pcn_pl] = gts_pcn
        feed_dict[prob_octants_pl] = missing_octants
        feed_dict[is_training_pl] = True

        train_op_output, loss, inputs_check, fine_output, summary = sess.run([train_op, model.loss,
                                                                              model.inputs_check,
                                                                              model.outputs,
                                                                              train_summary], feed_dict=feed_dict)
        # print ('dropout_ features 0:', dropout_features[:, 0, :])
        writer.add_summary(summary, step)

        if args.batch_check == True and step == 100:
            os.makedirs(args.results_dir, exist_ok=True)
            np.savez(os.path.join(args.results_dir, 'batch_check'),
                     inputs_check = inputs_check)
            log_string(LOG_FOUT, 'batch check saved')
            print ('batch check saved')

        if step % args.steps_per_visu_save == 0 or step == args.max_step:
            if os.path.exists(args.results_dir) == False:
                os.makedirs(args.results_dir)

            np.savez(os.path.join(args.results_dir, 'results_{}'.format(step)),
                     partial_input_octants = sep_inputs,
                     complete_output = fine_output, inputs_lmdb = inputs_octants,
                     npts_octants = npts_octs, ids = ids,
                     missing_octants = prob,
                     inputs_pcn = inputs_pcn, npts_pcn = npts_pcn, gts_pcn = gts_pcn)

        if step % args.steps_per_eval == 0 and step != 0:
            df_valid, num_valid = lmdb_dataflow(args.lmdb_valid, args.batch_size,
                                        args.num_input_points_octants, args.num_gt_points_octants,
                                        args.num_input_points_pcn, args.num_input_points_pcn,
                                        args.fuzzy_boundary, is_training=True)

            valid_gen = df_valid.get_data()

            num_eval_steps = num_valid // args.batch_size

            chamfer_l = []
            for e in range(num_eval_steps):

                ids, inputs_octants, npts_octs, gts_octants, missing_octants, \
                inputs_pcn, npts_pcn, gts_pcn, gt_original_pcn = next(train_gen)

                # missing_octants: B x 8: what all octants of each pc in B are all zero
                for i in range(args.batch_size):
                    if np.sum(gts_octants[:, i, :, :]**2) == 0:
                # if gt.shape[1] < args.batch_size:
                        print('batch incomplete. skipping', gts_octants.shape, args.batch_size)
                        continue

                npts_octant = np.sum(npts_octs, axis=1) # (8,)
                inputs_org_octants = np.split(inputs_octants[0], np.cumsum(npts_octant)[:-1])

                sep_inputs = []
                for i in range(8):
                    sep_inputs.append(np.split(inputs_org_octants[i], np.cumsum(npts_octs[i])[:-1]))

                feed_dict = {}
                for i in range(8):
                    # feed_dict[inputs_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
                    feed_dict[inputs_octants_pl[i]] = np.expand_dims(inputs_org_octants[i], axis=0)
                    feed_dict[npts_octants_pl[i]] = npts_octs[i]
                    feed_dict[gt_octants_pl[i]] = gts_octants[i]
                    feed_dict[missing_octants_pl] = missing_octants
                    # print ('Shape here:', i, inputs_org_octants[i].shape)

                feed_dict[inputs_pcn_pl] = inputs_pcn
                feed_dict[npts_pcn_pl] = npts_pcn
                feed_dict[gt_pcn_pl] = gts_pcn
                feed_dict[prob_octants_pl] = missing_octants
                feed_dict[is_training_pl] = True

                fine_output = sess.run(model.outputs, feed_dict=feed_dict)
                fine_output = np.array(fine_output)

                res = np.concatenate(fine_output[0], axis=1)
                output_concat = np.concatenate((res, fine_output[1]), axis=1)
                # import ipdb; ipdb.set_trace()
                for bv in range(args.batch_size):
                    gt_points = gt_original_pcn[bv].reshape(-1, 3)
                    output_points = output_concat[bv, :, :].reshape(-1, 3)


                    (min_dist_0_np, idx_0_np) = sess.run([min_dist, min_idx],
                                                     feed_dict={source_pc: gt_points,
                                                               target_pc: output_points})

                    (min_dist_1_np, idx_1_np) = sess.run([min_dist, min_idx],
                                                             feed_dict={source_pc: output_points,
                                                                       target_pc: gt_points})

                    chamfer_val = np.sum([np.mean(min_dist_1_np), np.mean(min_dist_0_np)]) * 100
                    chamfer_l.append(chamfer_val)

            print ('Final mean:', np.mean(chamfer_l))

            summary = tf.Summary(value=[tf.Summary.Value(tag="Chamfer Distance",
                                                 simple_value=np.mean(chamfer_l))])
            writer.add_summary(summary, step)

        if step % args.steps_per_print == 0:
            log_string(LOG_FOUT, 'step %d  loss %.8f' % (step, loss))
            print('step %d  loss %.8f' % (step, loss))

        if step % args.steps_per_save == 0 or step == args.max_step:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            log_string(LOG_FOUT, 'Model saved at %s' % args.log_dir)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet/valid.lmdb')
    parser.add_argument('--log_restore_dir', default='log/pcn_emd')
    parser.add_argument('--log_dir', default='log/pcn_emd')
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=1500)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=10000)
    parser.add_argument('--steps_per_visu', type=int, default=3000)
    parser.add_argument('--steps_per_save', type=int, default=100000)
    parser.add_argument('--visu_freq', type=int, default=5)
    parser.add_argument('--num_parts', type=int, default=8)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--steps_per_visu_save', type=int, default=100000)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--encoder_dropout', type=float, default=0.5)
    parser.add_argument('--prob_type', type=str)
    parser.add_argument('--prob_zero_start', type=int, default=1)
    parser.add_argument('--prob_zero_end', type=int, default=4)
    parser.add_argument('--fuzzy_boundary', type=float, default=0)
    parser.add_argument('--num_input_points_octants', type=int, default=1500)
    parser.add_argument('--num_gt_points_octants', type=int, default=16384)
    parser.add_argument('--num_input_points_pcn', type=int, default=1500)
    parser.add_argument('--num_gt_points_pcn', type=int, default=16384)
    parser.add_argument('--lmdb_test', default='data/shapenet/valid.lmdb')
    parser.add_argument('--batch_check', action='store_true')
    parser.add_argument('--slurm_output_file', type=str)
    parser.add_argument('--gt_to_pred_weight', type=float, default=1.0)
    parser.add_argument('--pred_to_gt_weight', type=float, default=0.0)

    args = parser.parse_args()

    if args.train == True:
        train(args)

    elif args.valid == True:
        valid(args)

    elif args.test == True:
        test(args)