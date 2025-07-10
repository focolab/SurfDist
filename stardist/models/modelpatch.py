from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import math
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tqdm import tqdm


from csbdeep.models import BaseConfig
from csbdeep.internals.blocks import conv_block3, unet_block, resnet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage
from distutils.version import LooseVersion
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from skimage.measure  import regionprops
from skimage.filters.rank import minimum, maximum
keras = keras_import()
K = keras_import('backend')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard, CSVLogger, ModelCheckpoint = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard', 'CSVLogger', 'ModelCheckpoint')
Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Concatenate = keras_import('layers', 'Input', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Add', 'Concatenate')
Model = keras_import('models', 'Model')
from tensorflow.keras import activations
from tensorflow.keras.layers import Activation
from tensorflow.keras.saving import get_custom_objects
from .base import StarDistBase, StarDistDataBase, _tf_version_at_least
from .base import generic_masked_loss as base_generic_masked_loss, masked_loss as base_masked_loss, masked_loss_mae as base_masked_loss_mae
from ..sample_patches import sample_patches
from ..utils import edt_prob, _normalize_grid, mask_to_categorical
from ..matching import relabel_sequential
from ..geometry import patch_dist, mesh_to_label
from ..rays3d import Rays_Patch, rays_from_json, Rays_GoldenSpiral
from ..nms import non_maximum_suppression_patch, non_maximum_suppression_patch_sparse
from ..bezier_utils import icosahedron, subdivide_tri_tf, subdivide_tri_tf_paged, subdivide_tri_tf_precomputed, tf_safe_norm, tf_safe_norm_three, dists_to_controls_tf, eval_triangles_tf, pred_instances_to_control_points, subdivide_tris_tf_bary_one_shot

best_norm = np.inf

@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32), tf.TensorSpec(shape=[None,None,3], dtype=tf.float32), tf.TensorSpec(shape=[None,4], dtype=tf.float32), tf.TensorSpec(shape=[3], dtype=tf.float32),))
def tf_raydist(labeled_instances, unit_directions, starting_points, grid=tf.constant((1.,1.,1.))):
    starting_points *= tf.concat((tf.constant((1.,)), grid), 0)
    starting_points = tf.broadcast_to(starting_points[...,None,:], tf.concat((tf.shape(starting_points)[:-1], tf.stack((tf.shape(unit_directions)[-2], tf.shape(starting_points)[-1]))), 0))
    padded_unit_directions = tf.concat((tf.zeros(tf.concat((tf.shape(unit_directions)[:-1], tf.constant((1,))), 0), dtype=tf.float32), unit_directions), axis=-1)
    starting_values = tf.gather_nd(labeled_instances, tf.cast(tf.math.round(starting_points), tf.int32))
    next_values = tf.identity(starting_values)
    current_points = tf.identity(starting_points)
    next_points = tf.identity(starting_points)
    rounded_next_points = tf.cast(next_points, tf.int32)
    lower_bounds_float = tf.zeros_like(current_points)
    upper_bounds = tf.cast(tf.shape(labeled_instances),tf.int32)[None,None,...] - 1
    upper_bounds_float = tf.cast(upper_bounds, tf.float32)
    upper_bounds_float = tf.broadcast_to(upper_bounds_float, tf.shape(starting_points))
    not_finished = tf.ones(tf.shape(starting_points)[:-1], dtype=tf.bool)
    not_finished_float = tf.cast(not_finished, tf.float32)
    not_finished_float_mask = tf.stack((not_finished_float, not_finished_float, not_finished_float, not_finished_float), -1)
    in_bounds = tf.ones_like(not_finished)
    not_hit_background = tf.ones_like(not_finished)
    while tf.math.reduce_any(not_finished):
        next_points = current_points + padded_unit_directions
        in_bounds = tf.math.reduce_any(next_points >= lower_bounds_float, axis=-1)
        in_bounds &= tf.math.reduce_any(next_points <= upper_bounds_float, axis=-1)
        if not tf.math.reduce_all(in_bounds):
            not_finished |= in_bounds
            not_finished_float = tf.cast(not_finished, tf.float32)
            not_finished_float_mask = tf.stack((not_finished_float, not_finished_float, not_finished_float, not_finished_float), -1)
            padded_unit_directions *= not_finished_float_mask
            next_points = current_points + padded_unit_directions

        next_points = tf.math.round(next_points)
        rounded_next_points = tf.cast(next_points, tf.int32)
        next_values = tf.gather_nd(labeled_instances, rounded_next_points)
        not_hit_background = next_values == starting_values
        if not tf.math.reduce_all(not_hit_background):
            not_finished &= not_hit_background
            not_finished_float = tf.cast(not_finished, tf.float32)
            not_finished_float_mask = tf.stack((not_finished_float, not_finished_float, not_finished_float, not_finished_float), -1)
            padded_unit_directions *= not_finished_float_mask

        current_points += padded_unit_directions
    diffs = current_points[...,1:] - starting_points[...,1:]
    dists = tf_safe_norm_three(diffs)
    tf.debugging.assert_all_finite(dists,"tf_rd")
    # add 1 to get dist to background (rather than last foreground)
    return dists + 1.

def tf_get_surface_points(labeled_instances):
    foreground_points = labeled_instances[:,1:-1,1:-1,1:-1] != 0
    surface_points = labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,:-2,1:-1,1:-1]
    surface_points |= labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,2:,1:-1,1:-1]
    surface_points |= labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,1:-1,:-2,1:-1]
    surface_points |= labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,1:-1,2:,1:-1]
    surface_points |= labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,1:-1,1:-1,:-2]
    surface_points |= labeled_instances[:,1:-1,1:-1,1:-1] != labeled_instances[:,1:-1,1:-1,2:]
    surface_points = tf.cast(surface_points, labeled_instances.dtype) * labeled_instances[:,1:-1,1:-1,1:-1]
    return tf.pad(surface_points, ((0,0), (1,1), (1,1), (1,1)), constant_values=0)

def cdist_batched(points_1, points_2):
    points_1 = points_1.reshape((points_1.shape[0],-1,3))
    points_2 = points_2.reshape((points_2.shape[0],-1,3))
    # num_points = points_1.shape[0]
    # points_1_length = points_1.shape[1]
    # points_2_length = points_2.shape[1]
    # points_1 = tf.broadcast_to(points_1, [points_2_length, num_points, points_1_length, 3])
    # points_1 = tf.transpose(points_1, [1,2,0,3])
    # points_2 = tf.broadcast_to(points_2, [points_1_length, num_points, points_2_length, 3])
    # points_2 = tf.transpose(points_2, [1,0,2,3])
    # return tf.keras.losses.cosine_similarity(points_1,points_2, axis=-1)
    diff = points_1[:,:,None,:] - points_2[:,None,:,:]
    diff = np.square(diff)
    diff = np.sum(diff, axis=-1)
    dists = np.sqrt(diff)
    return dists
def cdist_batched_paged(points_1, points_2):
    batch_size = 100
    dists = np.zeros((points_1.shape[0], points_1.shape[1], points_2.shape[1]))
    def get_batch_cdist(start_index, dists, points_1_, points_2_):
        stop_index = np.minimum(start_index + batch_size, len(points_1))
        batch_indices = np.arange(start_index, stop_index)
        batch_points_1 = points_1_[batch_indices]
        batch_points_2 = points_2_[batch_indices]
        start_index += batch_size
        batch_dists = cdist_batched(batch_points_1, batch_points_2)
        dists[batch_indices] = batch_dists
        return start_index, dists, points_1, points_2
    start_index = 0
    while start_index < len(dists):
        start_index, dists, points_1, points_2 = get_batch_cdist(start_index, dists, points_1, points_2)
    return dists

def tf_cdist_batched(points_1, points_2):
    points_1 = tf.reshape(points_1, (points_1.shape[0],-1,3))
    points_2 = tf.reshape(points_2, (points_2.shape[0],-1,3))
    diff = points_1[...,None,:] - points_2[:,None,...]
    diff = tf.math.square(diff)
    diff = tf.math.reduce_sum(diff, axis=-1)
    dists = tf.math.sqrt(diff)
    return dists

def tf_cdist_batched_paged(points_1, points_2):
    batch_size = tf.constant(10000)
    dists = tf.zeros((points_1.shape[0], points_1.shape[1], points_2.shape[1]))
    def get_batch_cdist(start_index, dists, points_1_, points_2_):
        stop_index = tf.minimum(start_index + batch_size, len(points_1))
        batch_indices = tf.range(start_index, stop_index)
        batch_points_1 = tf.gather(points_1_, batch_indices)
        batch_points_2 = tf.gather(points_2_, batch_indices)
        start_index += batch_size
        batch_dists = tf_cdist_batched(batch_points_1, batch_points_2)
        dists = tf.tensor_scatter_nd_update(dists, tf.expand_dims(batch_indices, -1), batch_dists)
        return start_index, dists, points_1, points_2
    def not_done(start_index, dists, points_1_, points_2_):
        return tf.less(start_index, len(dists))
    start_index = tf.constant(0)
    start_index, dists, points_1, points_2 = tf.while_loop(not_done, get_batch_cdist, (start_index, dists, points_1, points_2))
    tf.debugging.assert_all_finite(dists,"tf_cd_b_p")
    return dists

def tf_chamfer_dist(points_1, points_2):
    batch_size = tf.constant(5000)
    p1_to_p2 = tf.zeros(points_1.shape[:-1], points_1.dtype)
    p2_to_p1 = tf.zeros(points_2.shape[:-1], points_2.dtype)
    def get_batch_cdist(start_index, p1_to_p2, p2_to_p1, points_1_, points_2_):
        stop_index = tf.minimum(start_index + batch_size, len(points_1))
        batch_indices = tf.range(start_index, stop_index)
        batch_points_1 = tf.gather(points_1_, batch_indices)
        batch_points_2 = tf.gather(points_2_, batch_indices)
        start_index += batch_size
        batch_dists = tf_cdist_batched(batch_points_1, batch_points_2)
        p1_to_p2 = tf.tensor_scatter_nd_update(p1_to_p2, tf.expand_dims(batch_indices, -1), tf.math.reduce_min(batch_dists, axis=-1))
        p2_to_p1 = tf.tensor_scatter_nd_update(p2_to_p1, tf.expand_dims(batch_indices, -1), tf.math.reduce_min(batch_dists, axis=-2))
        return start_index, p1_to_p2, p2_to_p1, points_1, points_2
    def not_done(start_index, p1_to_p2, p2_to_p1, points_1_, points_2_):
        return tf.less(start_index, len(p1_to_p2))
    start_index = tf.constant(0)
    start_index, p1_to_p2, p2_to_p1, points_1, points_2 = tf.while_loop(not_done, get_batch_cdist, (start_index, p1_to_p2, p2_to_p1, points_1, points_2))
    tf.debugging.assert_all_finite(p1_to_p2,"tf_c_d")
    tf.debugging.assert_all_finite(p2_to_p1,"tf_c_d")
    return p1_to_p2, p2_to_p1

def interpolate_surfaces(surface_points, N_rays):
    def interpolate(points):
        surface_points_norms = tf_safe_norm(points)
        surface_points_norms = tf.broadcast_to(surface_points_norms[...,None], points.shape)
        unit_surface_points = tf.math.divide_no_nan(points, surface_points_norms)
        unit_n_rays = N_rays.vertices_tf
        unit_n_rays = tf.broadcast_to(unit_n_rays, (unit_surface_points.shape[0],) + unit_n_rays.shape)
        dists = tf_cdist_batched_paged(unit_n_rays, unit_surface_points)
        three_nearest_dists, three_nearest_dist_args = tf.math.top_k(-dists, k=3)
        three_nearest_dists *= -1
        three_nearest_surface_points = tf.gather(points, three_nearest_dist_args, batch_dims=1)
        three_nearest_surface_point_dists = tf_safe_norm(three_nearest_surface_points)
        is_closest_surface_point = three_nearest_surface_point_dists == tf.math.reduce_min(three_nearest_surface_point_dists, axis=-1)[...,None]
        weights = 1. / tf.math.square(three_nearest_dists)
        infinite_weights = ~tf.math.is_finite(weights)
        if tf.math.count_nonzero(infinite_weights) > 0:
            infinite_weight_rows = tf.math.reduce_any(infinite_weights, axis=-1)
            multiple_infinite_weight_rows = tf.math.reduce_any(tf.math.greater(tf.cast(infinite_weights, tf.float32), 1), axis=-1)
            infinite_weight_row_updates = tf.zeros(weights[infinite_weight_rows].shape)
            weights = tf.tensor_scatter_nd_update(weights, tf.where(infinite_weight_rows), infinite_weight_row_updates)
            infinite_weights &= tf.math.logical_or(
                ~multiple_infinite_weight_rows[...,None],
                is_closest_surface_point)
            weights += tf.cast(infinite_weights, tf.float32)
        n_interpolated_surface_point_dists = weights * three_nearest_surface_point_dists
        n_interpolated_surface_point_dists = tf.math.reduce_mean(n_interpolated_surface_point_dists, axis=-1)
        weight_sums = tf.math.reduce_sum(weights, axis=-1)
        n_interpolated_surface_point_dists = tf.math.divide_no_nan(n_interpolated_surface_point_dists, weight_sums/3)
        n_interpolated_surface_points = unit_n_rays * n_interpolated_surface_point_dists[...,None]
        return n_interpolated_surface_points
    
    origin_included = tf.math.reduce_any(tf.math.reduce_all(surface_points==0., axis=-1), axis=-1)
    no_origin_points = surface_points[~origin_included]
    origin_points = surface_points[origin_included]
    interpolated_surface_points = None
    if len(origin_points) > 0:
        # origin_rows = tf.math.reduce_all(origin_points==0., axis=-1)
        # origin_points = origin_points[~origin_rows]
        # origin_points = tf.RaggedTensor.from_row_lengths(origin_points, tf.math.reduce_sum(tf.cast(~origin_rows, tf.int32), axis=-1))
        interpolated_origin_points = interpolate(tf.identity(origin_points))
        interpolated_surface_points = tf.zeros((surface_points.shape[0], interpolated_origin_points.shape[1], 3))
        interpolated_surface_points = tf.tensor_scatter_nd_update(interpolated_surface_points, tf.where(origin_included), interpolated_origin_points)
    if len(no_origin_points) > 0:
        interpolated_no_origin_points = interpolate(tf.identity(no_origin_points))
        if interpolated_surface_points is None:
            interpolated_surface_points = tf.zeros((surface_points.shape[0], interpolated_no_origin_points.shape[1], 3))
        interpolated_surface_points = tf.tensor_scatter_nd_update(interpolated_surface_points, tf.where(~origin_included), interpolated_no_origin_points)
    return interpolated_surface_points

def tf_simdist_batched(points_1, points_2):
    points_1 = tf.reshape(points_1, (points_1.shape[0],-1,3))
    points_2 = tf.reshape(points_2, (points_2.shape[0],-1,3))
    num_points = points_1.shape[0]
    points_1_length = points_1.shape[1]
    points_2_length = points_2.shape[1]
    points_1 = tf.broadcast_to(points_1, [points_2_length, num_points, points_1_length, 3])
    points_1 = tf.transpose(points_1, [1,2,0,3])
    points_2 = tf.broadcast_to(points_2, [points_1_length, num_points, points_2_length, 3])
    points_2 = tf.transpose(points_2, [1,0,2,3])
    return -tf.keras.losses.cosine_similarity(points_1,points_2, axis=-1)

# surface_points: M x N x 3
# N_rays: K x 3
# return: M x K x 3
def intimidate(surface_points, N_rays):
    is_origin = tf.math.reduce_all(surface_points==0.,axis=-1)
    surface_points += K.epsilon() * tf.cast(is_origin[...,None], tf.float32)
    surface_points_norms = tf_safe_norm(surface_points)
    surface_points_norms = tf.broadcast_to(surface_points_norms[...,None], surface_points.shape)
    unit_surface_points = tf.math.divide_no_nan(surface_points, surface_points_norms)
    unit_n_rays = N_rays.vertices_tf
    unit_n_rays = tf.broadcast_to(unit_n_rays, (unit_surface_points.shape[0],) + unit_n_rays.shape)
    similarities = tf_simdist_batched(unit_n_rays, surface_points)

    three_nearest_sims, three_nearest_sim_args = tf.math.top_k(similarities, k=1)
    three_nearest_surface_points = tf.gather(surface_points, three_nearest_sim_args, batch_dims=1)
    three_nearest_surface_points_norms = tf_safe_norm(three_nearest_surface_points)
    near_enough = tf.cast(three_nearest_sims > .85, tf.float32)
    averaged_dists = three_nearest_surface_points_norms*near_enough
    averaged_dists = tf.math.reduce_sum(averaged_dists, axis=-1)
    averaged_dists = tf.math.divide_no_nan(averaged_dists, tf.math.reduce_sum(near_enough,axis=-1))

    # averaged_dists = three_nearest_surface_points_norms*three_nearest_sims
    # averaged_dists = tf.math.reduce_sum(averaged_dists, axis=-1)
    return averaged_dists[...,None]*unit_n_rays

def generic_masked_loss(mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=K.abs):
    def _loss(y_true, y_pred):
        actual_loss = K.mean(mask[...,None,:] * weights * loss(y_true, y_pred), axis=-1)
        norm_mask = (K.mean(mask) + K.epsilon()) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = K.mean((1-mask[...,None,:]) * reg_penalty(y_pred), axis=-1)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask
    return _loss

def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

# TODO: should we use norm_by_mask=True in the loss or only in a metric?
#       previous 2D behavior was norm_by_mask=False
#       same question for reg_weight? use 1e-4 (as in 3D) or 0 (as in 2D)?

def masked_loss_mae(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_mse(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.square, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_mae(mask):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, K.abs, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mae

def masked_metric_mse(mask):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, K.square, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mse

def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.binary_crossentropy(y_true, y_pred) - K.binary_crossentropy(y_true, y_true), axis=-1)


def masked_loss_iou(mask, reg_weight=0, norm_by_mask=True):
    def iou_loss(y_true, y_pred):
        axis = -1 if backend_channels_last() else 1
        # y_pred can be negative (since not constrained) -> 'inter' can be very large for y_pred << 0
        # - clipping y_pred values at 0 can lead to vanishing gradients
        # - 'K.sign(y_pred)' term fixes issue by enforcing that y_pred values >= 0 always lead to larger 'inter' (lower loss)
        inter = K.mean(K.sign(y_pred)*K.square(K.minimum(y_true,y_pred)), axis=axis)
        union = K.mean(K.square(K.maximum(y_true,y_pred)), axis=axis)
        iou = inter/(union+K.epsilon())
        iou = K.expand_dims(iou,axis)
        loss = 1. - iou # + 0.005*K.abs(y_true-y_pred)
        return loss
    return generic_masked_loss(mask, iou_loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_iou(mask, reg_weight=0, norm_by_mask=True):
    def iou_metric(y_true, y_pred):
        axis = -1 if backend_channels_last() else 1
        y_pred = K.maximum(0., y_pred)
        inter = K.mean(K.square(K.minimum(y_true,y_pred)), axis=axis)
        union = K.mean(K.square(K.maximum(y_true,y_pred)), axis=axis)
        iou = inter/(union+K.epsilon())
        loss = K.expand_dims(iou,axis)
        return loss
    return generic_masked_loss(mask, iou_metric, reg_weight=reg_weight, norm_by_mask=norm_by_mask)


def weighted_categorical_crossentropy(weights, ndim):
    """ ndim = (2,3) """

    axis = -1 if backend_channels_last() else 1
    shape = [1]*(ndim+2)
    shape[axis] = len(weights)
    weights = np.broadcast_to(weights, shape)
    weights = K.constant(weights)

    def weighted_cce(y_true, y_pred):
        # ignore pixels that have y_true (prob_class) < 0
        mask = K.cast(y_true>=0, K.floatx())
        y_pred /= K.sum(y_pred+K.epsilon(), axis=axis, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        loss = - K.sum(weights*mask*y_true*K.log(y_pred), axis = axis)
        return loss

    return weighted_cce

class StarDistDataPatch(StarDistDataBase):
    @staticmethod
    def get_surface(single_instance_mask):
        footprint = np.array([
            [[0,0,0],[0,1,0],[0,0,0]],
            [[0,1,0],[1,1,1],[0,1,0]],
            [[0,0,0],[0,1,0],[0,0,0]]
            ])
        return np.logical_and(minimum(single_instance_mask, footprint) == 0, single_instance_mask >= 1)

    @staticmethod
    def get_labeled_surfaces(labels):
        footprint = np.array([
            [[0,0,0],[0,1,0],[0,0,0]],
            [[0,1,0],[1,1,1],[0,1,0]],
            [[0,0,0],[0,1,0],[0,0,0]]
            ])
        foreground = labels != 0
        # minimum and maximum produce zero output with uint32 input (and possibly any other input type other than uint8 or uint16)
        mins = minimum(labels.astype(np.uint16), footprint)
        maxes = maximum(labels.astype(np.uint16), footprint)
        adjacent_other = np.logical_or(mins != labels, maxes != labels)
        return labels * np.logical_and(foreground, adjacent_other)

    def get_surfaces(self, labeled_instances, N, grid=(1,1,1)):


        grid = _normalize_grid(grid,3)
        labeled_instances = labeled_instances.astype(np.uint16, copy=False)
        dst_shape = tuple(s // a for a, s in zip(grid, labeled_instances.shape)) + (N,3)
        dst = np.empty(dst_shape, np.float32)

        downsampled_labeled_instances = labeled_instances[::grid[0],::grid[1],::grid[2]]

        voxelwise = np.zeros(downsampled_labeled_instances.shape + (N,3))
        z,y,x = np.indices(downsampled_labeled_instances.shape)
        voxelwise_indices = np.array((z,y,x)).transpose((1,2,3,0))
        for label in np.unique(labeled_instances):
            # ignore instances of less than three voxels (otherwise argpartition will fail)
            if label != 0 and np.sum(downsampled_labeled_instances==label) > 3:
                blob_mask = downsampled_labeled_instances == label
                surface = self.get_surface(blob_mask)
                surface_points = np.array(np.where(surface)).T

                blob_edt = edt_prob(blob_mask, anisotropy=self.anisotropy)
                blob_centroid = np.array(np.unravel_index(blob_edt.argmax(), blob_edt.shape))

                surface_point_directions = surface_points - blob_centroid[None,...]
                surface_point_directions = surface_point_directions.astype(float)
                surface_point_norms = np.linalg.norm(surface_point_directions, axis=-1)
                surface_point_norms = np.nan_to_num(surface_point_norms)
                surface_point_directions /= surface_point_norms[...,None]
                surface_point_directions = np.nan_to_num(surface_point_directions)
                unit_n_rays = self.N_rays.vertices
                dists = cdist(unit_n_rays, surface_point_directions)
                try:
                    three_nearest_dist_args = np.argpartition(dists, kth=3, axis=-1)
                except:
                    print(dists)
                    print(surface_point_directions)
                    print(dists.shape)
                    print(surface_point_directions.shape)
                    np.save("problem_labels.npy", labeled_instances)
                    raise Exception

                three_nearest_dist_args = three_nearest_dist_args[:,:3]
                three_nearest_dists = np.take_along_axis(dists, three_nearest_dist_args, axis=-1)
                three_nearest_surface_point_norms = np.take_along_axis(surface_point_norms, three_nearest_dist_args.reshape(-1), axis=-1)
                three_nearest_surface_point_norms = three_nearest_surface_point_norms.reshape((-1, 3))
                weights = 1. / np.square(three_nearest_dists)
                finite_weights = np.isfinite(weights)
                finite_rows_mask = np.sum(~finite_weights, axis=-1) == 0
                weights[~finite_weights] = 0.
                weights *= finite_rows_mask[...,None]
                weights += ~finite_weights
                n_interpolated_surface_point_dists = np.average(three_nearest_surface_point_norms, weights=weights, axis=-1)
                interpolated_surface_points = unit_n_rays * n_interpolated_surface_point_dists[...,None] + blob_centroid[None,...]
                translated_interpolated_surface_points = interpolated_surface_points[None,...] - voxelwise_indices[blob_mask][:,None,:]
                voxelwise[blob_mask] = translated_interpolated_surface_points
        voxelwise *= np.array(grid)
        assert(np.all(np.isfinite(voxelwise)))
        return voxelwise

    def __init__(self, X, Y, batch_size, rays, length,
                 n_classes=None, classes=None,
                 patch_size=(128,128,128), grid=(1,1,1), anisotropy=None, augmenter=None, foreground_prob=0, voronai=True, **kwargs):
        # TODO: support shape completion as in 2D?

        self.N = kwargs.pop("max_surface_points")
        self.num_subdivisions = 2
        self.num_subdivided_vertices = rays.cached_subdivision_output[self.num_subdivisions]['vertextofacemap_tf'].shape[0]

        super().__init__(X=X, Y=Y, n_rays=len(rays), grid=grid,
                         classes=classes, n_classes=n_classes,
                         batch_size=batch_size, patch_size=patch_size, length=length,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.rays = rays
        self.N_rays = Rays_GoldenSpiral(n=self.N, anisotropy=self.rays.anisotropy)
        self.voronai = voronai
        self.n_dist = len(self.rays)
        if self.voronai:
            self.n_dist *= 3
        self.anisotropy = anisotropy
        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'
        # re-use arrays
        if self.batch_size > 1:
            self.out_X = np.empty((self.batch_size,)+tuple(self.patch_size)+(() if self.n_channel is None else (self.n_channel,)), dtype=np.float32)
            patch_size_grid = tuple((p-1)//g+1 for p,g in zip(self.patch_size,self.grid))
            self.out_edt_prob = np.empty((self.batch_size,)+patch_size_grid, dtype=np.float32)
            self.out_patch_dist = np.empty((self.batch_size,)+patch_size_grid+(self.N,3), dtype=np.float32)
            self.out_label = np.empty((self.batch_size,)+tuple(self.patch_size)+(() if self.n_channel is None else (self.n_channel,)), dtype=np.float32)
            if self.n_classes is not None:
                self.out_prob_class = np.empty((self.batch_size,)+tuple(self.patch_size)+(self.n_classes+1,), dtype=np.float32)


    def __getitem__(self, i):
        idx = self.batch(i)
        arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                 patch_size=self.patch_size, n_samples=1,
                                 valid_inds=self.get_valid_inds(k)) for k in idx]

        if self.n_channel is None:
            X, Y = list(zip(*[(x[0],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1), y[0]) for y,*x in arrays]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X,Y))))

        if len(Y) == 1:
            X = X[0][np.newaxis]
        else:
            X = np.stack(X, out=self.out_X[:len(Y)])
        if X.ndim == 4: # input image has no channel axis
            X = np.expand_dims(X,-1)

        tmp = [edt_prob(lbl, anisotropy=self.anisotropy)[self.ss_grid[1:]] for lbl in Y]
        if len(Y) == 1:
            prob = tmp[0][np.newaxis]
        else:
            prob = np.stack(tmp, out=self.out_edt_prob[:len(Y)])

        # tmp = [patch_dist(lbl, self.rays, mode=self.sd_mode, grid=self.grid) for lbl in Y]
        # below not needed for no distances. temporarily disable to check whether it's killing epoch times with large batches
        # tmp = [self.get_surfaces(lbl, self.N, grid=self.grid) for lbl in Y]
        # if len(Y) == 1:
        #     dist = tmp[0][np.newaxis]
        # else:
        #     dist = np.stack(tmp, out=self.out_patch_dist[:len(Y)])

        prob = dist_mask = np.expand_dims(prob, -1)

        # append dist_mask to dist as additional channel
        # dist = dist.reshape(dist.shape[:-2]+(-1,))
        # dist = np.concatenate([dist,dist_mask],axis=-1)

        tmp = [lbl for lbl in Y]
        if len(Y) == 1:
            label = tmp[0][np.newaxis]
        else:
            label = np.stack(tmp, out=self.out_label[:len(Y)])
        label = np.expand_dims(label, -1)

        assert(np.all(np.isfinite(X)))
        assert(np.all(np.isfinite(prob)))
        # assert(np.all(np.isfinite(dist)))
        assert(np.all(np.isfinite(label)))
        if self.n_classes is None:
            return [X], [prob, label]
        else:
            tmp = [mask_to_categorical(y, self.n_classes, self.classes[k]) for y,k in zip(Y, idx)]
            # TODO: downsample here before stacking?
            if len(Y) == 1:
                prob_class = tmp[0][np.newaxis]
            else:
                prob_class = np.stack(tmp, out=self.out_prob_class[:len(Y)])

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(prob_class, (1,)+tuple(1/g for g in self.grid)+(1,), order=0)

            return [X], [prob, label, prob_class]



class ConfigPatch(BaseConfig):
    """Configuration for a :class:`StarDist3D` model.

    Parameters
    ----------
    axes : str or None
        Axes of the input images.
    rays : Rays_Base, int, or None
        Ray factory (e.g. Ray_GoldenSpiral).
        If an integer then Ray_GoldenSpiral(rays) will be used
    n_channel_in : int
        Number of channels of given input image (default: 1).
    grid : (int,int,int)
        Subsampling factors (must be powers of 2) for each of the axes.
        Model will predict on a subsampled grid for increased efficiency and larger field of view.
    n_classes : None or int
        Number of object classes to use for multi-class predection (use None to disable)
    anisotropy : (float,float,float)
        Anisotropy of objects along each of the axes.
        Use ``None`` to disable only for (nearly) isotropic objects shapes.
        Also see ``utils.calculate_extents``.
    backbone : str
        Name of the neural network architecture to be used as backbone.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_pool : (int,int,int)
        Maxpooling size for all (U-Net) convolution layers.
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    unet_* : *
        Additional parameters for U-net backbone.
    resnet_n_blocks : int
        Number of ResNet blocks.
    resnet_kernel_size : (int,int,int)
        Convolution kernel size for all ResNet blocks.
    resnet_n_filter_base : int
        Number of convolution kernels (feature channels) for ResNet blocks.
        (Number is doubled after every downsampling, see ``grid``.)
    net_conv_after_resnet : int
        Number of filters of the extra convolution layer after ResNet (0 to disable).
    resnet_* : *
        Additional parameters for ResNet backbone.
    train_patch_size : (int,int,int)
        Size of patches to be cropped from provided training images.
    train_background_reg : float
        Regularizer to encourage distance predictions on background regions to be 0.
    train_foreground_only : float
        Fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels.
    train_sample_cache : bool
        Activate caching of valid patch regions for all training images (disable to save memory for large datasets)
    train_dist_loss : str
        Training loss for star-convex polygon distances ('mse' or 'mae').
    train_loss_weights : tuple of float
        Weights for losses relating to (probability, distance)
    train_epochs : int
        Number of training epochs.
    train_steps_per_epoch : int
        Number of parameter update steps per epoch.
    train_learning_rate : float
        Learning rate for training.
    train_batch_size : int
        Batch size for training.
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress.
    train_n_val_patches : int
        Number of patches to be extracted from validation images (``None`` = one patch per image).
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
    use_gpu : bool
        Indicate that the data generator should use OpenCL to do computations on the GPU.

        .. _ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    """

    def __init__(self, axes='ZYX', rays=None, n_channel_in=1, grid=(1,1,1), n_classes=None, anisotropy=None, backbone='unet', voronai=True, **kwargs):

        if rays is None:
            if 'rays_json' in kwargs:
                rays = rays_from_json(kwargs['rays_json'])
            elif 'n_rays' in kwargs:
                rays = Rays_Patch(kwargs['n_rays'])
            else:
                rays = Rays_Patch(12)
        elif np.isscalar(rays):
            rays = Rays_Patch(rays)
        
        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1+len(rays))

        # directly set by parameters
        self.n_rays                    = len(rays)
        self.grid                      = _normalize_grid(grid,3)
        self.anisotropy                = anisotropy if anisotropy is None else tuple(anisotropy)
        self.backbone                  = str(backbone).lower()
        self.rays_json                 = rays.to_json()
        self.n_classes                 = None if n_classes is None else int(n_classes)
        self.voronai = voronai
        self.n_dist = len(rays)
        self.n_faces = len(rays.faces)
        self.n_edges = len(rays.edges)
        self.predict_dirs = False
        if self.voronai:
            self.n_dist *= 3

        if 'anisotropy' in self.rays_json['kwargs']:
            if self.rays_json['kwargs']['anisotropy'] is None and self.anisotropy is not None:
                self.rays_json['kwargs']['anisotropy'] = self.anisotropy
                print("Changing 'anisotropy' of rays to %s" % str(anisotropy))
            elif self.rays_json['kwargs']['anisotropy'] != self.anisotropy:
                warnings.warn("Mismatch of 'anisotropy' of rays and 'anisotropy'.")

        # default config (can be overwritten by kwargs below)
        if self.backbone == 'unet':
            self.unet_n_depth            = 2
            self.unet_kernel_size        = 3,3,3
            self.unet_n_filter_base      = 32
            self.unet_n_conv_per_depth   = 2
            self.unet_pool               = 2,2,2
            self.unet_activation         = 'relu'
            self.unet_last_activation    = 'relu'
            self.unet_batch_norm         = False
            self.unet_dropout            = 0.0
            self.unet_prefix             = ''
            self.net_conv_after_unet     = 128
        elif self.backbone == 'resnet':
            self.resnet_n_blocks         = 4
            self.resnet_kernel_size      = 3,3,3
            self.resnet_kernel_init      = 'he_normal'
            self.resnet_n_filter_base    = 32
            self.resnet_n_conv_per_block = 3
            self.resnet_activation       = 'relu'
            self.resnet_batch_norm       = False
            self.net_conv_after_resnet   = 128
        else:
            raise ValueError("backbone '%s' not supported." % self.backbone)

        # net_mask_shape not needed but kept for legacy reasons
        if backend_channels_last():
            self.net_input_shape       = None,None,None,self.n_channel_in
            self.net_mask_shape        = None,None,None,1
        else:
            self.net_input_shape       = self.n_channel_in,None,None,None
            self.net_mask_shape        = 1,None,None,None

        # self.train_shape_completion    = False
        # self.train_completion_crop     = 32
        self.train_patch_size          = 128,128,128
        self.train_background_reg      = 1e-4
        self.train_foreground_only     = 0.9
        self.train_sample_cache        = True

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = (1,0.1) if self.n_classes is None else (1,0.1,1)
        self.train_class_weights       = (1,1) if self.n_classes is None else (1,)*(self.n_classes+1)
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 100
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 1
        self.train_n_val_patches       = None
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'min_delta'
        self.train_reduce_lr           = {'factor': 0.5, 'patience': 100, min_delta_key: 0}

        self.use_gpu                   = False

        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out', 'n_rays', 'rays_json'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        # FIXME: put into is_valid()
        if not len(self.train_loss_weights) == (2 if self.n_classes is None else 3):
            raise ValueError(f"train_loss_weights {self.train_loss_weights} not compatible with n_classes ({self.n_classes}): must be 3 weights if n_classes is not None, otherwise 2")

        if not len(self.train_class_weights) == (2 if self.n_classes is None else self.n_classes+1):
            raise ValueError(f"train_class_weights {self.train_class_weights} not compatible with n_classes ({self.n_classes}): must be 'n_classes + 1' weights if n_classes is not None, otherwise 2")


class PatchDist(StarDistBase):
    """StarDist3D model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=ConfigPatch(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        if self.config.backbone == "unet":
            return self._build_unet()
        elif self.config.backbone == "resnet":
            return self._build_resnet()
        else:
            raise NotImplementedError(self.config.backbone)


    def _build_unet(self):
        assert self.config.backbone == 'unet'
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv3D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling3D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        def relu2pi(x):
            return activations.sigmoid(x) * tf.constant([2*np.pi])
        def swishplus(x):
            return activations.swish(x) + tf.constant([0.2785])
        get_custom_objects().update({'relu2pi': Activation(relu2pi), 'swishplus': Activation(swishplus)})

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(unet)
        if self.config.predict_dirs:
            output_voronai_theta = Conv3D(self.config.n_rays, (1,1,1), name='theta', padding='same', activation='relu2pi')(unet)
            output_unet_theta = Concatenate(axis=-1, name='phi_input')([unet, output_voronai_theta])
            output_voronai_phi = Conv3D(self.config.n_rays, (1,1,1), name='phi', padding='same', activation='sigmoid')(output_unet_theta)
            output_unet_theta_phi = Concatenate(axis=-1, name='dist_input')([unet, output_voronai_theta, output_voronai_phi])
            output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='softplus')(output_unet_theta_phi)
            output_b111_barycentric = Conv3D(2*self.config.n_faces, (1,1,1), name='b_111_bary', padding='same', activation='sigmoid')(output_unet_theta_phi)
            output_unet_all_directions = Concatenate(axis=-1, name='b_other_dists_input')([output_unet_theta_phi, output_b111_barycentric])
            output_non_vertex_control_dists = Conv3D(2*self.config.n_edges + self.config.n_faces, (1,1,1), name='b_other_dists', padding='same', activation='softplus')(output_unet_all_directions)
            output_surface = Concatenate(axis=-1, name='surface')([output_dist, output_voronai_theta, output_voronai_phi, output_b111_barycentric, output_non_vertex_control_dists])
        else:
            output_dist = Conv3D(self.config.n_rays + 2*self.config.n_edges + self.config.n_faces, (1,1,1), name='pre_dist', padding='same', activation='linear')(unet)
            output_surface = Conv3D(self.config.n_rays + 2*self.config.n_edges + self.config.n_faces, (1,1,1), name='dist', padding='same', activation='softplus')(output_dist)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_surface,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_surface])


    def _build_resnet(self):
        assert self.config.backbone == 'resnet'
        n_filter = self.config.resnet_n_filter_base
        resnet_kwargs = dict (
            kernel_size        = self.config.resnet_kernel_size,
            n_conv_per_block   = self.config.resnet_n_conv_per_block,
            batch_norm         = self.config.resnet_batch_norm,
            kernel_initializer = self.config.resnet_kernel_init,
            activation         = self.config.resnet_activation,
        )

        input_img = Input(self.config.net_input_shape, name='input')

        layer = input_img
        layer = Conv3D(n_filter, (7,7,7), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)
        layer = Conv3D(n_filter, (3,3,3), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)

        pooled = np.array([1,1,1])
        for n in range(self.config.resnet_n_blocks):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            if any(p > 1 for p in pool):
                n_filter *= 2
            layer = resnet_block(n_filter, pool=tuple(pool), **resnet_kwargs)(layer)

        layer_base = layer

        if self.config.net_conv_after_resnet > 0:
            layer = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                           name='features', padding='same', activation=self.config.resnet_activation)(layer_base)

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(layer)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='swish')(layer)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_resnet > 0:
                layer_class  = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                                      name='features_class', padding='same', activation=self.config.resnet_activation)(layer_base)
            else:
                layer_class  = layer_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(layer_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])

    def get_max_surface_points(self, truth_sets):
        grid = self.config.grid
        max_surface_points = 0
        for truth_set in truth_sets:
            for y in truth_set:
                labels = np.unique(y)
                labeled_surfaces = StarDistDataPatch.get_labeled_surfaces(y)
                downsampled_labeled_surfaces = labeled_surfaces[::grid[0],::grid[1],::grid[2]]
                surface_labels, surface_label_counts = np.unique(downsampled_labeled_surfaces, return_counts=True)
                surface_label_counts[surface_labels==0] = -1
                max_label_count = np.max(surface_label_counts)
                if max_label_count > max_surface_points:
                    max_surface_points = max_label_count
        return max_surface_points
        
    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        if optimizer is None:
            optimizer = Adam(self.config.train_learning_rate)#, clipnorm=0.001)#, clipvalue=100)

        masked_dist_loss = {'mse': masked_loss_mse,
                            'mae': masked_loss_mae,
                            'iou': masked_loss_iou,
                            }[self.config.train_dist_loss]
        prob_loss = 'binary_crossentropy'


        def split_dist_true_mask(dist_true_mask):
            return tf.split(dist_true_mask, num_or_size_splits=[self.data_train.N*3,-1], axis=-1)

        def dist_pred_to_residual_loss(true_labels, dist_pred):
            grid = self.config.grid
            true_labels = tf.squeeze(true_labels)
            downsampled_true_labels = true_labels[:,::grid[0],::grid[1],::grid[2]]
            dist_mask = downsampled_true_labels != 0
            dist_mask = tf.squeeze(dist_mask)

            # convert distances to surfaces
            rays = rays_from_json(self.config.rays_json)
            if self.config.predict_dirs:
                split_sizes = self.config.n_rays, self.config.n_rays, self.config.n_rays, 2*self.config.n_faces, 2*self.config.n_edges + self.config.n_faces
                dists_pred, thetas_pred, phis_pred, b111_bary_pred, other_control_dists_pred = tf.split(dist_pred, num_or_size_splits=split_sizes, axis=-1)
            else:
                split_sizes = self.config.n_rays, 2*self.config.n_edges + self.config.n_faces
                dists_pred, other_control_dists_pred = tf.split(dist_pred, num_or_size_splits=split_sizes, axis=-1)
            # dists_pred = tf.stop_gradient(dists_pred)
            dists_pred_uncorrected = tf.identity(dists_pred)
            dists_pred *= tf.cast(dists_pred >= 0, tf.float32)
            dist_pred = dists_pred

            other_control_dists_pred_uncorrected = tf.identity(other_control_dists_pred)
            other_control_dists_pred *= tf.cast(other_control_dists_pred >= 0, tf.float32)

            valid_inds = tf.where(dist_mask)
            valid_dist_pred = tf.gather_nd(dist_pred, valid_inds)

            if self.config.predict_dirs:
                valid_thetas_pred = tf.gather_nd(thetas_pred,valid_inds)
                valid_phis_pred = tf.gather_nd(phis_pred,valid_inds)
                valid_cartesian_vertex_directions = rays.voronai_vertices_to_unit_vertices_tf(valid_thetas_pred, valid_phis_pred)
            else:
                valid_cartesian_vertex_directions = tf.broadcast_to(rays.vertices_tf, (tf.math.reduce_prod(valid_dist_pred.shape[:-1]), self.config.n_rays, 3))
                valid_cartesian_vertex_directions = tf.reshape(valid_cartesian_vertex_directions, tuple(valid_dist_pred.shape[:-1]) + (self.config.n_rays, 3))
            # print("min pred:", tf.math.reduce_min(tf.gather_nd(dist_pred,valid_inds)), "max pred:", tf.math.reduce_max(tf.gather_nd(dist_pred,valid_inds)), "avg pred", tf.math.reduce_mean(tf.gather_nd(dist_pred,valid_inds)))

            starting_points = tf.cast(valid_inds, tf.float32)

            tf.debugging.assert_all_finite(dist_pred, "Encountered NaN in dist_pred")
            # subdivide to sub-voxel resolution
            faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = rays.faces, rays.edges, rays.facetoedgemap, rays.vertextofacemap, rays.facetoedgesign

            valid_cartesian_vertices = tf.squeeze(valid_cartesian_vertex_directions)*valid_dist_pred[...,None]
            valid_other_control_dists_pred = tf.gather_nd(other_control_dists_pred, valid_inds)
            valid_other_control_dists_pred_uncorrected = tf.gather_nd(other_control_dists_pred_uncorrected, valid_inds)
            valid_other_control_dists_pred_negative_sums = tf.math.reduce_sum(tf.math.abs(valid_other_control_dists_pred_uncorrected) * tf.cast(valid_other_control_dists_pred_uncorrected < 0, tf.float32), axis=-1)
            if self.config.predict_dirs:
                valid_b111_barys = tf.gather_nd(b111_bary_pred, valid_inds)
            else:
                valid_b111_barys = tf.constant(((np.nan,np.nan),), tf.float32)
            valid_all_controls = pred_instances_to_control_points(valid_cartesian_vertex_directions, valid_dist_pred, valid_other_control_dists_pred, valid_b111_barys, rays.edges_tf, rays.faces_tf, rays.facetoedgemap_tf, rays.facetoedgesign_tf)

            # do bidirectional matching to calculate residuals
            # predicted_surface_points = np.array(np.where(predicted_surface)).T
            # blob_surface_points = np.array(np.where(blob_surface)).T
            def mesh_not_subdivided(cartesian_vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions):
                edge_vertices = tf.gather_nd(cartesian_vertices, tf.expand_dims(edges, axis=-1))
                max_edge_len = tf.math.reduce_max(tf_safe_norm(edge_vertices[:,0]-edge_vertices[:,1]))
                return tf.math.less(subdivisions, 1)
                # return tf.logical_and(subdivisions < 4, tf.math.greater(max_edge_len, 1))
            # residual_loss = tf.zeros(dist_mask.shape + (self.data_train.N + self.data_train.num_subdivided_vertices,))
            residual_loss = tf.zeros(dist_mask.shape + (self.data_train.num_subdivided_vertices,))

            def subdivision_loss(input_points):
                true_surface_points, predicted_surface_points, predicted_indices, predicted_controls = input_points
                predicted_surface_points_shape = predicted_surface_points.shape
                num_predicted_blobs = predicted_surface_points.shape[0]
                faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = rays.faces_tf, rays.edges_tf, rays.facetoedgemap_tf, rays.vertextofacemap_tf, rays.facetoedgesign_tf
                subdivision_counter = 0
                unsubbed = predicted_surface_points

                # TEMP TESTING: sample instead of subdivide
                unit_dists = tf.ones_like(ray_dists)
                # control_points = dists_to_controls_tf(predicted_surface_points, faces, unit_dists, vertextofacemap)
                # center_points = eval_triangles_tf(predicted_controls)
                # predicted_surface_points = tf.concat((predicted_surface_points, center_points), axis=-2)
                while subdivision_counter < self.data_train.num_subdivisions:
                    predicted_surface_points, _, subdivision_counter = subdivide_tri_tf_precomputed(predicted_surface_points, rays, subdivision_counter, controls=predicted_controls)

                # predicted_surface_points = subdivide_tri_tf_paged(predicted_surface_points, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, 2)

                true_to_pred, pred_to_true = tf_chamfer_dist(true_surface_points, predicted_surface_points)
                pred_to_true_normalizer = true_to_pred.shape[-1] / pred_to_true.shape[-1]
                pred_to_true *= pred_to_true_normalizer
                chamfer = tf.concat((true_to_pred, pred_to_true), axis=-1)
                return chamfer

            @tf.function(input_signature=(tf.TensorSpec(shape=[None, self.config.n_rays, 3], dtype=tf.float32), tf.TensorSpec(shape=[None, self.config.n_faces, 10, 3], dtype=tf.float32), tf.TensorSpec(shape=[None, 4], dtype=tf.float32), tf.TensorSpec(shape=(None,)+self.config.train_patch_size, dtype=tf.float32),))
            def raydist_loss(predicted_surface_points, predicted_controls, starting_points, true_labels):
                addl_predicted_surface_points = subdivide_tris_tf_bary_one_shot(predicted_controls, rays.cached_subdivision_output[self.data_train.num_subdivisions]['all_addl_bary_unsubbed_faces'], rays.cached_subdivision_output[self.data_train.num_subdivisions]['all_addl_bary_vertices'])
                predicted_surface_points = tf.concat((predicted_surface_points, addl_predicted_surface_points), axis=-2)
                predicted_surface_point_norms = tf_safe_norm_three(predicted_surface_points)
                predicted_surface_point_directions = predicted_surface_points / predicted_surface_point_norms[...,None]
                true_ray_dists = tf_raydist(true_labels, predicted_surface_point_directions, starting_points, grid=tf.constant(grid, dtype=tf.float32))
                return tf.math.abs(true_ray_dists - predicted_surface_point_norms)

            # raydist loss
            addl_predicted_surface_points = subdivide_tris_tf_bary_one_shot(valid_all_controls, rays.cached_subdivision_output[self.data_train.num_subdivisions]['all_addl_bary_unsubbed_faces'], rays.cached_subdivision_output[self.data_train.num_subdivisions]['all_addl_bary_vertices'])
            predicted_surface_points = tf.concat((valid_cartesian_vertices, addl_predicted_surface_points), axis=-2)
            predicted_surface_point_norms = tf_safe_norm_three(predicted_surface_points)
            predicted_surface_point_directions = predicted_surface_points / predicted_surface_point_norms[...,None]
            true_ray_dists = tf_raydist(true_labels, predicted_surface_point_directions, starting_points, grid=tf.constant(grid, dtype=tf.float32))
            masked_residual_loss = tf.math.abs(true_ray_dists - predicted_surface_point_norms)

            # input_points = (valid_true_surface_points,valid_cartesian_vertices,valid_inds, valid_all_controls)
            # masked_residual_loss = subdivision_loss(input_points)

            # input_tuple = (valid_cartesian_vertices, valid_all_controls, starting_points, true_labels)
            # masked_residual_loss = raydist_loss(valid_cartesian_vertices, valid_all_controls, starting_points, true_labels)
            masked_residual_loss += valid_other_control_dists_pred_negative_sums[...,None]

            residual_loss = tf.tensor_scatter_nd_update(residual_loss, valid_inds, masked_residual_loss)

            invalid_inds = tf.where(not dist_mask)
            invalid_vertex_dists = tf.gather_nd(dists_pred_uncorrected, invalid_inds)
            invalid_other_dists = tf.gather_nd(other_control_dists_pred_uncorrected, invalid_inds)
            invalid_dist_sums = tf.math.reduce_sum(np.abs(invalid_vertex_dists), axis=-1) + tf.math.reduce_sum(np.abs(invalid_other_dists), axis=-1)
            regularized_invalid_dist_sums = invalid_dist_sums / tf.cast(invalid_vertex_dists.shape[-1] + invalid_other_dists.shape[-1], tf.float32)
            regularized_invalid_dist_sums = tf.broadcast_to(regularized_invalid_dist_sums[...,None], (regularized_invalid_dist_sums.shape[0] , residual_loss.shape[-1]))
            residual_loss = tf.tensor_scatter_nd_update(residual_loss, invalid_inds, regularized_invalid_dist_sums)

            residual_loss = tf.expand_dims(residual_loss, -1)
            tf.debugging.assert_all_finite(residual_loss,"dptrl")
            return residual_loss

        def surface_loss(labels, dist_pred):
            residual_loss = dist_pred_to_residual_loss(labels, dist_pred)
            tf.debugging.assert_all_finite(residual_loss,"sl")
            grid = self.config.grid
            downsampled_labels = labels[:,::grid[0],::grid[1],::grid[2]]
            dist_mask = downsampled_labels != 0
            dist_mask = tf.cast(dist_mask, tf.float32)
            return masked_dist_loss(dist_mask, reg_weight=self.config.train_background_reg)(tf.zeros_like(residual_loss), residual_loss)

        def dist_loss(true_labels, dist_pred):
            true_labels = tf.squeeze(true_labels)
            split_sizes = self.config.n_rays, self.config.n_rays, self.config.n_rays, 2*self.config.n_faces, 2*self.config.n_edges + self.config.n_faces
            dists_pred, thetas_pred, phis_pred, b111_bary_pred, other_control_dists_pred = tf.split(dist_pred, num_or_size_splits=split_sizes, axis=-1)
            thetas_pred, phis_pred, b111_bary_pred, other_control_dists_pred = tf.stop_gradient(thetas_pred), tf.stop_gradient(phis_pred), tf.stop_gradient(b111_bary_pred), tf.stop_gradient(other_control_dists_pred)
            grid = self.config.grid
            downsampled_true_labels = true_labels[:,::grid[0],::grid[1],::grid[2]]
            label_mask = downsampled_true_labels != 0
            valid_indices = tf.where(label_mask)
            starting_points = tf.cast(valid_indices, tf.float32)
            valid_thetas_pred = tf.gather_nd(thetas_pred,valid_indices)
            valid_phis_pred = tf.gather_nd(phis_pred,valid_indices)
            rays = rays_from_json(self.config.rays_json)
            valid_cartesian_directions = rays.voronai_vertices_to_unit_vertices_tf(valid_thetas_pred, valid_phis_pred)
            ray_dists = tf_raydist(downsampled_true_labels, valid_cartesian_directions, starting_points, grid=grid)
            ray_dists_true = tf.zeros(dists_pred.shape)
            ray_dists_true = tf.tensor_scatter_nd_update(ray_dists_true, valid_indices, ray_dists)
            label_mask = tf.broadcast_to(label_mask[...,None], ray_dists_true.shape)
            label_mask = tf.cast(label_mask, tf.float32)
            distance_loss = base_masked_loss_mae(label_mask, reg_weight=self.config.train_background_reg)(ray_dists_true, dists_pred)
            tf.debugging.assert_all_finite(distance_loss,"dl")
            return distance_loss

        def dist_iou_metric(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_iou(dist_mask, reg_weight=0)(dist_true, dist_pred)

        def relevant_mae(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mae(dist_mask)(dist_true, dist_pred)

        def relevant_mse(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mse(dist_mask)(dist_true, dist_pred)


        if self._is_multiclass():
            prob_class_loss = weighted_categorical_crossentropy(self.config.train_class_weights, ndim=self.config.n_dim)
            loss = [prob_loss, surface_loss, prob_class_loss]
        else:
            loss = [prob_loss, surface_loss]

        self.keras_model.compile(optimizer, loss         = loss,
                                            loss_weights = list(self.config.train_loss_weights),
                                            metrics      = {},
                                            run_eagerly=True)

        self.callbacks = []

        # class myCallback(tf.keras.callbacks.Callback):
        #     def on_epoch_begin(self, epoch, logs=None):
        #         print("starting epoch",epoch)
        #         print(self.model.trainable_variables)
        #     def on_epoch_end(self, epoch, logs=None):
        #         print("ending epoch",epoch)
        #         print(self.model.trainable_variables)
        # my_callback = myCallback()
        # self.callbacks.append(my_callback)

        csv_logger_callback = CSVLogger('training.log')
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='training_checkpoint.model.keras',save_best_only=True)
        self.callbacks.extend([csv_logger_callback, model_checkpoint_callback])
        class ClearSessionCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                K.clear_session()
        self.callbacks.append(ClearSessionCallback())
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=False))
                else:
                    pass
                    # self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0,ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def train(self, X, Y, validation_data, classes='auto', augmenter=None, seed=None, epochs=None, steps_per_epoch=None, workers=1):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Input images
        Y : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Label masks
        classes (optional): 'auto' or iterable of same length as X
             label id -> class id mapping for each label mask of Y if multiclass prediction is activated (n_classes > 0)
             list of dicts with label id -> class id (1,...,n_classes)
             'auto' -> all objects will be assigned to the first non-background class,
                       or will be ignored if config.n_classes is None
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`) or triple (if multiclass)
            Tuple (triple if multiclass) of X,Y,[classes] validation data.
        augmenter : None or callable
            Function with expected signature ``xt, yt = augmenter(x, y)``
            that takes in a single pair of input/label image (x,y) and returns
            the transformed images (xt, yt) for the purpose of data augmentation
            during training. Not applied to validation images.
            Example:
            def simple_augmenter(x,y):
                x = x + 0.05*np.random.normal(0,1,x.shape)
                return x,y
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        classes = self._parse_classes_arg(classes, len(X))

        if not self._is_multiclass() and classes is not None:
            warnings.warn("Ignoring given classes as n_classes is set to None")

        isinstance(validation_data,(list,tuple)) or _raise(ValueError())
        if self._is_multiclass() and len(validation_data) == 2:
            validation_data = tuple(validation_data) + ('auto',)
        ((len(validation_data) == (3 if self._is_multiclass() else 2))
            or _raise(ValueError(f'len(validation_data) = {len(validation_data)}, but should be {3 if self._is_multiclass() else 2}')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        div_by = self._axes_div_by(axes)
        [p % d == 0 or _raise(ValueError(
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            rays             = rays_from_json(self.config.rays_json),
            grid             = self.config.grid,
            patch_size       = self.config.train_patch_size,
            anisotropy       = self.config.anisotropy,
            use_gpu          = self.config.use_gpu,
            foreground_prob  = self.config.train_foreground_only,
            n_classes        = self.config.n_classes,
            sample_ind_cache = self.config.train_sample_cache,
        )

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = self._parse_classes_arg(validation_data[2], n_data_val) if self._is_multiclass() else None
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        max_surface_points = self.get_max_surface_points([Y, validation_data[1]])
        data_kwargs.update({"max_surface_points": max_surface_points})
        _data_val = StarDistDataPatch(validation_data[0],validation_data[1], classes=classes_val, batch_size=n_take, length=1, **data_kwargs)
        data_val = _data_val[0]

        # expose data generator as member for general diagnostics
        self.data_train = StarDistDataPatch(X, Y, classes=classes, batch_size=self.config.train_batch_size,
                                         augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)

        if self.config.train_tensorboard:
            # only show middle slice of 3D inputs/outputs
            input_slices, output_slices = [[slice(None)]*5], [[slice(None)]*5,[slice(None)]*5]
            i = axes_dict(self.config.axes)['Z']
            channel = axes_dict(self.config.axes)['C']
            _n_in  = _data_val.patch_size[i] // 2
            _n_out = _data_val.patch_size[i] // (2 * (self.config.grid[i] if self.config.grid is not None else 1))
            input_slices[0][1+i] = _n_in
            output_slices[0][1+i] = _n_out
            output_slices[1][1+i] = _n_out
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            output_slices[1][1+channel] = slice(0,(self.config.n_rays//_n)*_n, self.config.n_rays//_n)
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)]*5]
                output_slices[2][1+channel] = slice(1,1+(self.config.n_classes//_n)*_n, self.config.n_classes//_n)

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.input_slices = input_slices
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*5,None]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks):
                self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=data_val, log_dir=str(self.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, input_slices=input_slices, output_slices=output_slices))

        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(iter(self.data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      workers=workers, use_multiprocessing=workers>1,
                      callbacks=self.callbacks, verbose=2,
                      # set validation batchsize to training batchsize (only works in tf 2.x)
                      **(dict(validation_batch_size = self.config.train_batch_size) if _tf_version_at_least("2.2.0") else {}))
        self._training_finished()

        return history


    def _instances_from_prediction(self, img_shape, prob, dist, points=None, prob_class=None, prob_thresh=None, nms_thresh=None, overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi  class prediction
        """
        print("doing predictions")
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)

        # sparse prediction
        return_meshes = nms_kwargs.get('return_meshes', False)
        if points is not None:
            nms_result = non_maximum_suppression_patch_sparse(dist, prob, points, rays, img_shape, nms_thresh=nms_thresh, **nms_kwargs)
            if return_meshes:
                points, probi, disti, indsi, meshes = nms_result
            else:
                points, probi, disti, indsi = nms_result
            if prob_class is not None:
                prob_class = prob_class[indsi]

        # dense prediction
        else:
            nms_result = non_maximum_suppression_patch(dist, prob, rays, grid=self.config.grid,
                                                              prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if return_meshes:
                points, probi, disti, meshes  = nms_result
            else:
                points, probi, disti = nms_result
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5,Z=1.0):
            #   1. re-scale points (origins of polyhedra)
            #   2. re-scale vectors of rays object (computed from distances)
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale and 'Z' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X', 'Y', and 'Z'")
            rescale = (1/scale['Z'],1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,3)
            rays = rays.copy(scale=rescale)
        else:
            rescale = (1,1,1)

        if return_labels:
            labels = mesh_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label, verbose=verbose)

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        else:
            labels = None

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            # build the list of class ids per label via majority vote
            # zoom prob_class to img_shape
            # prob_class_up = zoom(prob_class,
            #                      tuple(s2/s1 for s1, s2 in zip(prob_class.shape[:3], img_shape))+(1,),
            #                      order=0)
            # class_id, label_ids = [], []
            # for reg in regionprops(labels):
            #     m = labels[reg.slice]==reg.label
            #     cls_id = np.argmax(np.mean(prob_class_up[reg.slice][m], axis = 0))
            #     class_id.append(cls_id)
            #     label_ids.append(reg.label)
            # # just a sanity check whether labels where in sorted order
            # assert all(x <= y for x,y in zip(label_ids, label_ids[1:]))
            # res_dict.update(dict(classes = class_id))
            # res_dict.update(dict(labels = label_ids))
            # self.p = prob_class_up

            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))
        if return_meshes:
            return labels, res_dict, meshes
        else:
            return labels, res_dict


    def _axes_div_by(self, query_axes):
        if self.config.backbone == "unet":
            query_axes = axes_check_and_normalize(query_axes)
            assert len(self.config.unet_pool) == len(self.config.grid)
            div_by = dict(zip(
                self.config.axes.replace('C',''),
                tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
            ))
            return tuple(div_by.get(a,1) for a in query_axes)
        elif self.config.backbone == "resnet":
            grid_dict = dict(zip(self.config.axes.replace('C',''), self.config.grid))
            return tuple(grid_dict.get(a,1) for a in query_axes)
        else:
            raise NotImplementedError()


    @property
    def _config_class(self):
        return ConfigPatch
