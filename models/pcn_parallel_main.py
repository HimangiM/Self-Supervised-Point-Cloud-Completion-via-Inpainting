from tf_util import *


class Model:
    def __init__(self, inputs_octant, npts_octant, gt_octant, alpha, num_parts_octant, prob_octant,
                 encoder_dropout_octant, missing_octants, inputs_pcn, npts_pcn, gt_pcn,
                 gt_to_pred_weight, pred_to_gt_weight):
        self.num_coarse = 128
        self.grid_size = 2 #4
        self.grid_scale = 0.05
        self.num_parts = num_parts_octant
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.num_coarse_pcn = 1024
        self.num_fine_pcn = self.grid_size ** 2 * self.num_coarse_pcn

        self.features_octant = self.create_encoder_octant(inputs_octant, npts_octant, missing_octants)
        self.dropout_features_octant, self.dropout_features_mlp_octant = \
                               self.create_dropout_features_octant(self.features_octant, encoder_dropout_octant)

        self.features_pcn = self.create_encoder_pcn(inputs_pcn, npts_pcn)

        self.encoder_features = self.octant_pcn_encoder_features(self.dropout_features_mlp_octant, self.features_pcn)

        self.encoder_features_one_hot = self.octant_pcn_encoder_one_hot(self.encoder_features)

        self.coarse_octant, self.fine_octant = self.create_decoder_octant(self.encoder_features_one_hot)

        self.coarse_pcn, self.fine_pcn = self.create_decoder_pcn(self.encoder_features)

        self.loss, self.inputs_check, self.update = self.create_loss(self.coarse_octant, self.fine_octant, gt_octant, alpha, prob_octant,
                                                  self.fine_pcn, gt_pcn, gt_to_pred_weight, pred_to_gt_weight)

        self.outputs = self.fine_octant, self.fine_pcn, self.coarse_pcn
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        # self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder_octant(self, inputs, npts, missing_octants):

        features_l = []
        for i in range(self.num_parts):
            with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
                features, _ = mlp_conv(inputs[i], [16, 32])
                features_global = point_unpool(point_maxpool(features, npts[i], keepdims=True), npts[i])
                features = tf.concat([features, features_global], axis=2)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
                features, _ = mlp_conv(features, [64, 128])
                features = point_maxpool(features, npts[i])

            features_l.append(features)

        features_tf = tf.stack((features_l[0], features_l[1], features_l[2],
                                 features_l[3], features_l[4], features_l[5],
                                 features_l[6], features_l[7]), axis=0)

        missing_octants_T = tf.transpose(missing_octants)
        return features_tf * tf.expand_dims(missing_octants_T, -1)

    def create_dropout_features_octant(self, features, encoder_dropout):
        features_T = tf.transpose(features, perm = [1, 2, 0])

        drop_features = tf.nn.dropout(features_T, keep_prob=encoder_dropout,
                                      noise_shape=[features_T.shape[0], 1,
                                      self.num_parts])
        # import ipdb; ipdb.set_trace()
        # print('Drop features here:', drop_features[:, 0, :])

        drop_features_mlp = mlp(drop_features, [1])
        drop_features_mlp_sq = tf.squeeze(drop_features_mlp, axis=2)

        return drop_features, drop_features_mlp_sq


    def create_encoder_pcn(self, inputs_pcn, npts_pcn):
        with tf.variable_scope('encoder_pcn_0', reuse=tf.AUTO_REUSE):
            features, _ = mlp_conv(inputs_pcn, [128, 256])
            features_global = point_unpool(point_maxpool(features, npts_pcn, keepdims=True), npts_pcn)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_pcn_1', reuse=tf.AUTO_REUSE):
            features, _ = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts_pcn)
        return features

    def octant_pcn_encoder_features(self, features_octant, features_pcn):

        with tf.variable_scope('feature_fusion', reuse=tf.AUTO_REUSE):
            feature_octant_upsample = mlp(features_octant, [1024], prefix='a_')
        return tf.maximum(feature_octant_upsample, features_pcn)

    def octant_pcn_encoder_one_hot(self, encoder_features):

        encoder_features_exp = tf.expand_dims(encoder_features, axis=2)
        encoder_features_exp_tile = tf.tile(encoder_features_exp, [1, 1, 8])
        centers_one_hot = tf.eye(8, batch_shape=[encoder_features_exp_tile.shape[0]])
        octant_features_one_hot = tf.concat((encoder_features_exp_tile, centers_one_hot), axis=1)

        return octant_features_one_hot

    def create_decoder_octant(self, features):

        coarse_l = []
        fine_l = []
        for i in range(self.num_parts):

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                coarse = mlp(features[:, :, i], [128, 128, self.num_coarse * 3])
                coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

            with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
                x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
                y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
                grid = tf.meshgrid(x, y)
                grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
                grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

                point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
                point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

                global_feat = tf.tile(tf.expand_dims(features[:, :, i], 1), [1, self.num_fine, 1])

                feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

                center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
                center = tf.reshape(center, [-1, self.num_fine, 3])

                fine = mlp_conv(feat, [64, 64, 3])[0] + center

            coarse_l.append(coarse)
            fine_l.append(fine)

        return coarse_l, fine_l

    def create_decoder_pcn(self, features):
        with tf.variable_scope('decoder_pcn', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [256, 256, self.num_coarse_pcn * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse_pcn, 3])

        with tf.variable_scope('folding_pcn', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse_pcn, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine_pcn, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine_pcn, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine_pcn, 3])

            fine = mlp_conv(feat, [128, 128, 3])[0] + center

        return coarse, fine

    def create_loss(self, coarse_octant, fine_octant, gt_octant, alpha, prob_octant,
                    coarse_pcn, gt_pcn, gt_to_pred_weight, pred_to_gt_weight):

        loss_octants_all = 0
        loss_coarse_all = 0
        loss_fine_all = 0
        loss_coarse_inter_batch_all = 0
        loss_fine_inter_batch_all = 0
        loss_combined = 0
        loss_pcn_all = 0

        # octant loss
        prob_stack = tf.unstack(prob_octant, axis=1)

        prob_exp = tf.expand_dims(prob_octant, axis=1)
        prob_exp_tile = tf.tile(prob_exp, [1, prob_exp.shape[0], 1])

        # Octant loss - 1D chamfer only - no earth mover
        fine_exp_re = tf.reshape(fine_octant, [8, -1, 4, fine_octant[0].shape[1], fine_octant[0].shape[2]])
        fine_exp_re_exp = tf.expand_dims(fine_exp_re[:, :, 0, :, :], axis=2)
        fine_exp_re_exp_tile = tf.tile(fine_exp_re_exp, [1, 1, 4, 1, 1])
        fine_exp_re_loss = tf.reshape(fine_exp_re_exp_tile, [-1, fine_octant[0].shape[1], fine_octant[0].shape[2]])


        gt_exp_re = tf.reshape(gt_octant, [8, -1, 4, gt_octant[0].shape[1], gt_octant[0].shape[2]])
        # gt_exp_re_exp = tf.expand_dims(gt_exp_re, axis=2)
        # gt_exp_re_exp_tile = tf.tile(gt_exp_re_exp, [1, 1, 4, 1, 1, 1])
        gt_exp_re_loss = tf.reshape(gt_exp_re, [-1, gt_octant[0].shape[1], gt_octant[0].shape[2]])


        prob_octant_T = tf.transpose(prob_octant)
        prob_re = tf.reshape(prob_octant_T, [8, -1, 4])
        # prob_exp_re_exp = tf.expand_dims(prob_re, axis=3)
        # prob_exp_re_exp_tile = tf.tile(prob_exp_re_exp, [1, 1, 1, 4])
        prob_exp_re_loss = tf.reshape(prob_re, [-1])

        loss_fine_inter_batch = chamfer_shapenet_weighted(gt_exp_re_loss,
                                                          fine_exp_re_loss,
                                                          prob_exp_re_loss,
                                                          gt_to_pred_weight,
                                                          pred_to_gt_weight)

        loss = alpha * loss_fine_inter_batch
        loss_octants_all += loss

        coarse_pcn_views = tf.reshape(coarse_pcn, [-1, 4, coarse_pcn.shape[1], coarse_pcn.shape[2]])
        coarse_pcn_first_view = tf.expand_dims(coarse_pcn_views[:, 0, :, :], axis=1)
        coarse_pcn_first_view_tile = tf.tile(coarse_pcn_first_view, [1, 4, 1, 1])
        coarse_pcn_first_view_re = tf.reshape(coarse_pcn_first_view_tile, [-1, coarse_pcn_first_view_tile.shape[2],
                                                                           coarse_pcn_first_view_tile.shape[3]])


        loss_pcn_all = chamfer_weighted(gt_pcn, coarse_pcn_first_view_re, gt_to_pred_weight, pred_to_gt_weight)

        # octants and pcn loss combined
        # octants_pred = tf.concat(fine_octant, axis=1)
        # combined_pred = tf.concat((octants_pred, coarse_pcn), axis=1)
        #
        # combined_pred_re = tf.reshape(combined_pred, [-1, 4, combined_pred.shape[1], combined_pred.shape[2]])
        # combined_pred_re_exp = tf.expand_dims(combined_pred_re[:, 0, :, :], axis=1)
        # combined_pred_re_exp_tile = tf.tile(combined_pred_re_exp, [1, 4, 1, 1])
        # combined_pred_re_loss = tf.reshape(combined_pred_re_exp_tile, [-1, combined_pred.shape[1], combined_pred.shape[2]])
        #
        # loss_chamfer_combined = chamfer_weighted(gt_pcn, combined_pred_re_loss, gt_to_pred_weight, pred_to_gt_weight)

        loss_all = loss_octants_all + loss_pcn_all

        add_train_summary('train/loss_octants', loss_octants_all)
        update_coarse = add_valid_summary('valid/loss_octants', loss_octants_all)

        add_train_summary('train/loss_pcn', loss_pcn_all)
        update_fine = add_valid_summary('valid/loss_pcn', loss_pcn_all)

        add_train_summary('train/loss', loss_all)
        update_loss = add_valid_summary('valid/loss', loss_all)

        return loss_all, [gt_exp_re_loss, fine_exp_re_loss, gt_pcn, coarse_pcn_first_view_re, gt_pcn], \
               [update_coarse, update_fine, update_loss]
