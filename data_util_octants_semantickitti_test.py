import numpy as np
import tensorflow as tf
from tensorpack import dataflow


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def resample_pcd_and_center(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    resampled_pcd = pcd[idx[:n]]
    # points_mean = np.mean(resampled_pcd, axis=0)
    # resampled_pcd = resampled_pcd - points_mean

    return resampled_pcd


def return_octants(points, bbox, pts_threshold=128, fuzzy_threshold=0):
    x_center, y_center, z_center = np.mean(bbox, 0)

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


class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size_octants, gt_size_octants,
                 input_size_pcn, gt_size_pcn,
                 fuzzy_boundary, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size_octants = input_size_octants
        self.gt_size_octants = gt_size_octants
        self.input_size_pcn = input_size_pcn
        self.gt_size_pcn = gt_size_pcn
        self.remainder = remainder
        self.use_list = use_list
        self.fuzzy_boundary = fuzzy_boundary

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)


    def _aggregate_batch(self, data_holder, use_list=False):
        ''' Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        '''
        '''
        we are taking the original lmdb given by authors
        x[2] -> complete -> all 8 octants
        octants <- x[2] [(N_1, 3), .., (N_8, 3)]
        resample_pcd(octants[i], self.input)
        inputs -> append everything
        
        gt -> reshape_pcd(octants[i], gt_size)
        
        '''
        # octants data
        # arr_idx = np.arange(self.batch_size)
        # arr_idx = np.reshape(arr_idx, (-1, 5))
        # arr_idx_T = arr_idx.T
        # np.random.shuffle(arr_idx_T)
        # idx_new = arr_idx_T.T[:, :4]
        # idx_data_holder = idx_new.flatten()

        ids = np.stack([x[-1] for e, x in enumerate(data_holder)])
        npts = [[] for _ in range(8)]
        resampled_octants = [[] for _ in range(8)]
        gts = [[] for _ in range(8)]
        missing_octants = []
        for e, x in enumerate(data_holder):
            octants = return_octants(x[0], bbox = x[1], pts_threshold=1, fuzzy_threshold=self.fuzzy_boundary)
            missing_octant_ = []
            for i, o in enumerate(octants):
                if o.shape[0] > self.input_size_octants:
                    resampled_octants[i].append(resample_pcd_and_center(o, self.input_size_octants))
                    npts[i].append(self.input_size_octants)
                else:
                    resampled_octants[i].append(o)
                    npts[i].append(o.shape[0])

                gts[i].append(resample_pcd_and_center(o, self.gt_size_octants))

                if np.sum(o**2) == 0:
                    missing_octant_.append(0)
                else:
                    missing_octant_.append(1)
            missing_octants.append(missing_octant_)


        resampled_octants_flattened = []
        npts_octants_flattened = []
        for i in range(8):
            octant_flattened = np.concatenate([x for x in resampled_octants[i]]).astype(np.float32) # N_1 x 3
            octant_npts = np.stack([x.shape[0] for x in resampled_octants[i]]).astype(np.int32) # B

            resampled_octants_flattened.append(octant_flattened)
            npts_octants_flattened.append(octant_npts)

        inputs_octants = np.expand_dims(np.concatenate([x for x in resampled_octants_flattened]),
                                0).astype(np.float32) # 1 x (N_1+N_2+..+N_8) x 3
        npts_octants = np.array(npts_octants_flattened).astype(np.int32) # 8 x B
        gts_octants = np.array(gts).astype(np.float32) # 8 X gt_size x 3

        missing_octants_octants = np.array(missing_octants).astype(np.int32) # B x 8

        # PCN data
        inputs_pcn = [resample_pcd(x[0], self.input_size_pcn) if x[0].shape[0] > self.input_size_pcn else x[0]
            for e, x in enumerate(data_holder)]
        inputs_pcn = np.expand_dims(np.concatenate([x for x in inputs_pcn]), 0).astype(np.float32)
        npts_pcn = np.stack([x[0].shape[0] if x[0].shape[0] < self.input_size_pcn else self.input_size_pcn
            for e, x in enumerate(data_holder)]).astype(np.int32)
        gts_pcn = np.stack([resample_pcd(x[0], self.gt_size_pcn) for e, x in enumerate(data_holder)]).astype(np.float32)
        gt_original_pcn = np.stack([resample_pcd(x[2], self.gt_size_pcn) for x in data_holder]).astype(np.float32)

        return ids, inputs_octants, npts_octants, gts_octants, missing_octants, \
               inputs_pcn, npts_pcn, gts_pcn, gt_original_pcn

        # inputs = [resample_pcd(x[1], self.input_size) if x[1].shape[0] > self.input_size else x[1]
        #     for x in data_holder]
        # inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        # npts = np.stack([x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
        #     for x in data_holder]).astype(np.int32)
        # gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        # return ids, inputs, npts, gts

def lmdb_dataflow_test(lmdb_path, batch_size, input_size_octants, output_size_octants,
                  input_size_pcn, output_size_pcn,
                  fuzzy_boundary, is_training, test_speed=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        # df = dataflow.LocallyShuffleData(df, buffer_size=2000)
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = BatchData(df, batch_size, input_size_octants, output_size_octants,
                   input_size_pcn, output_size_pcn, fuzzy_boundary)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=1)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size


def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {placeholder: value for placeholder, value in zip(placeholders, next(generator))}
    queue_runner = tf.contrib.training.FeedingQueueRunner(queue, [enqueue_op], close_op, feed_fns=[feed_fn])
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()
