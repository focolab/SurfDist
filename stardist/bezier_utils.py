# curved surface rendering routines

# reference: patent USA 6,462,738, Kato, Saul S., "Curved Surface Reconstruction"
# reference: https://en.wikipedia.org/wiki/Point-normal_triangle
# reference: https://en.wikipedia.org/wiki/B%C3%A9zier_curve

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from skimage.morphology import ball

# adapted from https://github.com/tensorflow/tensorflow/issues/12071#issuecomment-746112219
@tf.custom_gradient
def tf_safe_norm(x):
    y = tf.norm(x, axis=-1)
    def grad(dy):
        return dy[...,None] * (x / tf.keras.backend.maximum(y[...,None],tf.keras.backend.epsilon()))
    return y, grad
@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),))
@tf.custom_gradient
def tf_safe_norm_three(x):
    y = tf.norm(x, axis=-1)
    def grad(dy):
        return dy[...,None] * (x / tf.keras.backend.maximum(y[...,None],tf.keras.backend.epsilon()))
    return y, grad
@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),))
@tf.custom_gradient
def tf_safe_norm_four(x):
    y = tf.norm(x, axis=-1)
    def grad(dy):
        return dy[...,None] * (x / tf.keras.backend.maximum(y[...,None],tf.keras.backend.epsilon()))
    return y, grad

def normalize(vec):
    norm = np.sqrt(vec.dot(vec))
    vec = vec/norm if norm != 0 else vec  # avoid divide-by-zero
    return vec


def point_normal_interpolate(first_edge_vertices, second_edge_vertices, first_edge_vertex_normals, second_edge_vertex_normals, t):
    # compute point along the edge of a point-normal triangle
    # compute bezier control points from normals and points

    w32s = np.einsum("ij,ij->i", second_edge_vertices - first_edge_vertices, first_edge_vertex_normals)
    w23s = np.einsum("ij,ij->i", first_edge_vertices - second_edge_vertices, second_edge_vertex_normals)
    p0s = first_edge_vertices
    p3s = second_edge_vertices
    p2s = (1/3) * (2 * second_edge_vertices + first_edge_vertices - w32s[:,None] * second_edge_vertex_normals)
    p1s = (1/3) * (2 * first_edge_vertices + second_edge_vertices - w23s[:,None] * first_edge_vertex_normals)
    bezier_interps = ((1-t)**3)*p0s + 3*(1-t)*(1-t) * t*p1s + 3*(1-t)*t*t*p2s+(t**3)*p3s
    return bezier_interps

# compute vertex normals by averaging face normal directions
def get_vertex_normals(vertices, faces, vertextofacemap):
    # compute face normals
    face_normals = np.zeros([len(faces), 3])
    face_index = 0
    v0s = vertices[faces[:,0]]
    v1s = vertices[faces[:,1]]
    v2s = vertices[faces[:,2]]
    face_normals = np.cross(v1s - v0s, v2s - v1s)
    face_normals /= np.linalg.norm(face_normals, axis=-1)[:,None]
    # average face normal directions
    vertex_face_normals = []
    max_len = 0
    for vertex_index, v in enumerate(vertices):
        face_indices = vertextofacemap[vertex_index]
        these_face_normals = face_normals[face_indices]
        vertex_face_normals.append(these_face_normals)
        max_len = max(len(these_face_normals), max_len)
    vertex_face_normals_array = np.zeros((len(vertex_face_normals), max_len, 3), dtype=float)
    for vertex_i in range(len(vertex_face_normals)):
        vertex_normals = vertex_face_normals[vertex_i]
        vertex_face_normals_array[vertex_i][:len(vertex_normals)] = vertex_normals
    vertex_face_normal_sums = vertex_face_normals_array.sum(axis=1)
    vertex_face_normal_sum_norms = np.linalg.norm(vertex_face_normal_sums, axis=-1)
    vertex_normals = vertex_face_normal_sums / vertex_face_normal_sum_norms[:,None]
    return vertex_normals

def point_normal_interpolate_tf(first_edge_vertices, second_edge_vertices, first_edge_vertex_normals, second_edge_vertex_normals, t):
    # compute point along the edge of a point-normal triangle
    # compute bezier control points from normals and points
    w32s = tf.einsum("ij,ij->i", second_edge_vertices - first_edge_vertices, first_edge_vertex_normals)
    w23s = tf.einsum("ij,ij->i", first_edge_vertices - second_edge_vertices, second_edge_vertex_normals)
    p0s = first_edge_vertices
    p3s = second_edge_vertices
    p2s = (1/3) * (2 * second_edge_vertices + first_edge_vertices - tf.expand_dims(w32s, -1) * second_edge_vertex_normals)
    p1s = (1/3) * (2 * first_edge_vertices + second_edge_vertices - tf.expand_dims(w23s, -1) * first_edge_vertex_normals)
    bezier_interps = ((1-t)**3)*p0s + 3*(1-t)*(1-t) * t*p1s + 3*(1-t)*t*t*p2s+(t**3)*p3s
    return bezier_interps
# compute vertex normals by averaging face normal directions
def get_vertex_normals_tf(vertices, faces, vertextofacemap):
    # compute face normals
    face_normals = tf.zeros([len(faces), 3])
    face_index = 0
    v0s = tf.gather(vertices, tf.gather(faces, 0, axis=1))
    v1s = tf.gather(vertices, tf.gather(faces, 1, axis=1))
    v2s = tf.gather(vertices, tf.gather(faces, 2, axis=1))
    face_normals = tf.linalg.cross(v1s - v0s, v2s - v1s)
    face_normal_norms = tf_safe_norm(face_normals)
    face_normals /= tf.expand_dims(face_normal_norms, axis=-1)
    # average face normal directions
    vertex_face_normals = tf.gather(face_normals, vertextofacemap)
    vertex_face_normal_sums = tf.math.reduce_sum(vertex_face_normals, axis=1)
    vertex_face_normal_sum_norms = tf_safe_norm(vertex_face_normal_sums)
    # handle zero vectors
    vertex_face_normal_sum_norms += tf.cast(vertex_face_normal_sum_norms==0., vertex_face_normal_sum_norms.dtype)
    vertex_normals = vertex_face_normal_sums / tf.expand_dims(vertex_face_normal_sum_norms, axis=-1)
    return vertex_normals
def get_vertex_normals_tf_batched(vertices, faces, vertextofacemap):
    faces = tf.broadcast_to(faces[None,...], (len(vertices),)+faces.shape)
    face_normals = tf.zeros((len(vertices), len(faces), 3))
    v0s = tf.gather(vertices, tf.gather(faces, 0, axis=-1), axis=-2, batch_dims=1)
    v1s = tf.gather(vertices, tf.gather(faces, 1, axis=-1), axis=-2, batch_dims=1)
    v2s = tf.gather(vertices, tf.gather(faces, 2, axis=-1), axis=-2, batch_dims=1)
    face_normals = tf.linalg.cross(v1s - v0s, v2s - v1s)
    face_normal_norms = tf_safe_norm(face_normals)
    face_normals /= tf.expand_dims(face_normal_norms, axis=-1)
    # average face normal directions
    vertextofacemap = tf.tile(vertextofacemap[None,...], [len(vertices)]+[1 for dim in vertextofacemap.shape])
    vertex_face_normals = tf.gather(face_normals, vertextofacemap, axis=-2, batch_dims=1)
    vertex_face_normal_sums = tf.math.reduce_sum(vertex_face_normals, axis=-2)
    vertex_face_normal_sums = vertex_face_normal_sums.to_tensor()
    vertex_face_normal_sum_norms = tf_safe_norm(vertex_face_normal_sums)
    # handle zero vectors
    vertex_face_normal_sum_norms += tf.cast(vertex_face_normal_sum_norms==0., vertex_face_normal_sum_norms.dtype)
    vertex_normals = vertex_face_normal_sums / tf.expand_dims(vertex_face_normal_sum_norms, axis=-1)
    return vertex_normals

def subdivide_tri_tf_bary(controls, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face, return_barys=False):
    flattened_face_to_edge_map = tf.reshape(facetoedgemap, (-1))
    unique_e_is, unique_e_i_idx = tf.unique(flattened_face_to_edge_map)
    unique_first_idx = tf.math.unsorted_segment_min(tf.range(len(flattened_face_to_edge_map)), unique_e_i_idx, len(unique_e_is))
    unique_edge_face_indices = unique_first_idx//3
    unique_edge_face_edge_indices = unique_first_idx%3
    unique_edge_unsubbed_faces = tf.gather(bary_face_to_unsubbed_face, unique_edge_face_indices)
    unique_edge_control_points = tf.gather(controls, unique_edge_unsubbed_faces)
    unique_edge_bary_edge_faces = tf.gather(barycentric_edges, unique_edge_face_indices)
    unique_edge_bary_edges = tf.gather(unique_edge_bary_edge_faces, unique_edge_face_edge_indices, batch_dims=1)
    unique_edge_bary_means = tf.math.reduce_mean(unique_edge_bary_edges, axis=-2)    
    sort_order = tf.argsort(unique_e_is)
    unique_edge_bary_means = tf.gather(unique_edge_bary_means, sort_order)
    unique_edge_control_points = tf.gather(unique_edge_control_points, sort_order)
    new_vertices = eval_triangles_tf(all_control_points=unique_edge_control_points, s=unique_edge_bary_means[...,0], t=unique_edge_bary_means[...,1])
    if return_barys:
        return new_vertices, unique_edge_bary_means, tf.gather(unique_edge_unsubbed_faces, sort_order)
    else:
        return new_vertices

def subdivide_tri_tf_bary_one_shot(controls, rays, subdivisions):
    subdivided_controls = tf.gather(controls, rays.cached_subdivision_output[subdivisions]['all_addl_bary_unsubbed_faces'])
    s, t = tf.unstack(rays.cached_subdivision_output[subdivisions]['all_addl_bary_vertices'], axis=-1)
    return eval_triangles_tf(all_control_points=subdivided_controls, s=s, t=t)

def subdivide_tris_tf_bary(controls, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face):
    flattened_face_to_edge_map = tf.reshape(facetoedgemap, (-1))
    unique_e_is, unique_e_i_idx = tf.unique(flattened_face_to_edge_map)
    unique_first_idx = tf.math.unsorted_segment_min(tf.range(len(flattened_face_to_edge_map)), unique_e_i_idx, len(unique_e_is))
    unique_edge_face_indices = unique_first_idx//3
    unique_edge_face_edge_indices = unique_first_idx%3
    unique_edge_unsubbed_faces = tf.gather(bary_face_to_unsubbed_face, unique_edge_face_indices)
    unique_edge_control_points = tf.gather(controls, unique_edge_unsubbed_faces, axis=1)
    unique_edge_bary_edge_faces = tf.gather(barycentric_edges, unique_edge_face_indices)
    unique_edge_bary_edges = tf.gather(unique_edge_bary_edge_faces, unique_edge_face_edge_indices, batch_dims=1)
    unique_edge_bary_means = tf.math.reduce_mean(unique_edge_bary_edges, axis=-2)
    sort_order = tf.argsort(unique_e_is)
    unique_edge_bary_means = tf.gather(unique_edge_bary_means, sort_order)
    unique_edge_control_points = tf.gather(unique_edge_control_points, sort_order, axis=1)
    new_vertices = eval_triangles_tf(all_control_points=unique_edge_control_points, s=unique_edge_bary_means[...,0], t=unique_edge_bary_means[...,1])
    return new_vertices

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 10, 3], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None,2], dtype=tf.float32)))
def subdivide_tris_tf_bary_one_shot(controls, addl_bary_unsubbed_faces, addl_bary_vertices):
    subdivided_controls = tf.gather(controls, addl_bary_unsubbed_faces, axis=1)
    s, t = tf.unstack(addl_bary_vertices, axis=-1)
    return eval_triangles_tf_batch(all_control_points=subdivided_controls, s=s, t=t)


def subdivide_tri_tf_precomputed(vertices, rays, subdivisions, controls=None):
    before_subdivide_attrs = rays.cached_subdivision_output[subdivisions]
    faces, edges, vertextofacemap = before_subdivide_attrs['faces_tf'], before_subdivide_attrs['edges_tf'], before_subdivide_attrs['vertextofacemap_tf']

    if controls is not None:
        facetoedgemap = before_subdivide_attrs['facetoedgemap_tf']
        barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face = before_subdivide_attrs['barycentric_faces'], before_subdivide_attrs['barycentric_edges'], before_subdivide_attrs['bary_face_to_unsubbed_face']
        if controls.ndim==3:
            new_vertices = subdivide_tri_tf_bary(controls, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face)
        else:
            new_vertices = subdivide_tris_tf_bary(controls, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face)
    else:
        vertex_normals = get_vertex_normals_tf_batched(vertices, faces, vertextofacemap)

        edges = tf.broadcast_to(edges[None,...], (len(vertices),)+edges.shape)
        first_edge_vertices, second_edge_vertices = tf.gather(vertices, tf.gather(edges, 0, axis=-1), batch_dims=1), tf.gather(vertices, tf.gather(edges, 1, axis=-1), batch_dims=1)
        first_edge_vertex_normals, second_edge_vertex_normals = tf.gather(vertex_normals, tf.gather(edges, 0, axis=-1), batch_dims=1), tf.gather(vertex_normals, tf.gather(edges, 1, axis=-1), batch_dims=1)

        first_edge_vertices, second_edge_vertices = tf.reshape(first_edge_vertices, (-1,3)), tf.reshape(second_edge_vertices, (-1,3))
        first_edge_vertex_normals, second_edge_vertex_normals = tf.reshape(first_edge_vertex_normals, (-1,3)), tf.reshape(second_edge_vertex_normals, (-1,3))
        new_vertices = point_normal_interpolate_tf(first_edge_vertices, second_edge_vertices, first_edge_vertex_normals, second_edge_vertex_normals, 0.5)
        new_vertices = tf.reshape(new_vertices, (len(vertices), -1, 3))
    new_vertices = tf.concat((vertices, new_vertices), -2)
    return new_vertices, rays, subdivisions+1

def subdivide_tri_tf(vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions, barycentric_faces=None, barycentric_edges=None, bary_face_to_unsubbed_face=None, vertex_normals=None, just_compute_verts=False, refinement_level=1, control_points=None):

    new_vertices, new_faces, new_edges, new_facetoedgemap, new_vertextofacemap, new_facetoedgesign = None, None, None, None, None, None

    # sub_method can be linear or point-normal
    num_orig_vertices = len(vertices)

    # compute new vertices on edges
    e2nv = tf.zeros(len(edges), dtype=tf.int32)
    # point-normal triangles
    if vertex_normals is None:
        vertex_normals = get_vertex_normals_tf(vertices, faces, vertextofacemap)

    # (1) compute new vertices, one for each edge, and add to vertex list
    edge_indices = tf.range(len(edges))
    first_edge_vertices, second_edge_vertices = tf.gather(vertices, tf.gather(edges, 0, axis=1)), tf.gather(vertices, tf.gather(edges, 1, axis=1))
    first_edge_vertex_normals, second_edge_vertex_normals = tf.gather(vertex_normals, tf.gather(edges,0, axis=1)), tf.gather(vertex_normals, tf.gather(edges, 1, axis=1))
    if control_points is not None:
        if control_points.ndim==3:
            new_vertices = subdivide_tri_tf_bary(control_points, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face)
        else:
            new_vertices = subdivide_tris_tf_bary(control_points, facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face)
    else:
        new_vertices = point_normal_interpolate_tf(first_edge_vertices, second_edge_vertices, first_edge_vertex_normals, second_edge_vertex_normals, 0.5)
    new_vertices = tf.concat((vertices, new_vertices), 0)
    e2nv += len(new_vertices) - 1 - tf.range(len(edges), dtype=tf.int32)[::-1]

    new_faces = tf.zeros((len(faces)*4, 3), dtype=tf.int32)
    new_edges = tf.zeros((2*len(edges)+3*len(faces), 2), dtype=tf.int32)
    new_facetoedgemap = tf.zeros((len(new_faces), 3), dtype=tf.int32)
    E = len(edges)

    if not just_compute_verts:

        face_indices = tf.range(len(faces))
        face_edges = tf.gather(edges, facetoedgemap)
        v_1s, v_2s, v_3s = faces[:,0], faces[:,1], faces[:,2]
        v_1_in_edge = tf.cast(tf.math.reduce_sum(tf.cast(face_edges == v_1s[:,None,None], tf.uint8), axis=-1), tf.bool)
        v_2_in_edge = tf.cast(tf.math.reduce_sum(tf.cast(face_edges == v_2s[:,None,None], tf.uint8), axis=-1), tf.bool)
        v_3_in_edge = tf.cast(tf.math.reduce_sum(tf.cast(face_edges == v_3s[:,None,None], tf.uint8), axis=-1), tf.bool)
        e_i_1s = facetoedgemap[tf.math.logical_and(v_1_in_edge, v_2_in_edge)]
        e_i_2s = facetoedgemap[tf.math.logical_and(v_2_in_edge, v_3_in_edge)]
        e_i_3s = facetoedgemap[tf.math.logical_and(v_1_in_edge, v_3_in_edge)]

        v_4s, v_5s, v_6s = tf.gather(e2nv, e_i_1s), tf.gather(e2nv, e_i_2s), tf.gather(e2nv, e_i_3s)

        nf_i_bls = 4*face_indices
        nf_i_ts = 4*face_indices+1
        nf_i_brs = 4*face_indices+2
        nf_i_ms = 4*face_indices+3
        new_faces = tf.tensor_scatter_nd_update(new_faces, tf.expand_dims(nf_i_bls, -1), tf.stack((v_1s, v_4s, v_6s), axis=-1))
        new_faces = tf.tensor_scatter_nd_update(new_faces, tf.expand_dims(nf_i_ts, -1), tf.stack((v_2s, v_5s, v_4s), axis=-1))
        new_faces = tf.tensor_scatter_nd_update(new_faces, tf.expand_dims(nf_i_brs, -1), tf.stack((v_5s, v_6s, v_3s), axis=-1))
        new_faces = tf.tensor_scatter_nd_update(new_faces, tf.expand_dims(nf_i_ms, -1), tf.stack((v_4s, v_5s, v_6s), axis=-1))

        e_14s, e_42s, e_25s, e_53s, e_36s, e_61s, e_46s, e_45s, e_56s = tf.sort(
            tf.stack((
                (v_1s, v_4s, v_2s, v_5s, v_3s, v_6s, v_4s, v_4s, v_5s),
                (v_4s, v_2s, v_5s, v_3s, v_6s, v_1s, v_6s, v_5s, v_6s)),
            axis=-1),
        axis=-1)

        e_i_14s = 2*e_i_1s+1
        e_i_42s = 2*e_i_1s
        e_i_25s = 2*e_i_2s+1
        e_i_53s = 2*e_i_2s
        e_i_36s = 2*e_i_3s+1
        e_i_61s = 2*e_i_3s
        e_i_46s = tf.cast(2*E+3*face_indices, tf.int32)
        e_i_45s = tf.cast(2*E+3*face_indices+1, tf.int32)
        e_i_56s = tf.cast(2*E+3*face_indices+2, tf.int32)

        outer_e_is = tf.stack((e_i_14s, e_i_42s, e_i_25s, e_i_53s, e_i_36s, e_i_61s), axis=1)
        outer_e_is = tf.reshape(outer_e_is, (-1))
        outer_es = tf.stack((e_14s, e_42s, e_25s, e_53s, e_36s, e_61s), axis=1)
        outer_es = tf.reshape(outer_es, (-1, 2))

        unique_outer_e_is, unique_e_i_idx = tf.unique(outer_e_is)
        # get indices of first occurances of unique elements (adapted from https://stackoverflow.com/questions/48595802/tensorflow-finding-index-of-first-occurrence-of-elements-in-a-tensor)
        unique_first_idx = tf.math.unsorted_segment_min(tf.range(outer_e_is.shape[0]), unique_e_i_idx, unique_outer_e_is.shape[0])
        unique_outer_e_i_es = tf.gather(outer_es, unique_first_idx)
        new_edges = tf.scatter_nd(tf.expand_dims(unique_outer_e_is, -1), unique_outer_e_i_es, new_edges.shape)
        new_edges = tf.tensor_scatter_nd_update(new_edges,
            tf.expand_dims((e_i_46s, e_i_45s, e_i_56s), axis=-1),
            (e_46s, e_45s, e_56s))
        
        # index edges; tf.unique only supports 1d, and tf.uniquev2 doesn't support gradient -_-
        outer_es_1d = edges.shape[0] * outer_es[:,1] + outer_es[:,0]
        unique_outer_es_1d, unique_e_idx = tf.unique(outer_es_1d)
        outer_e_i_addends = tf.cast(unique_e_idx != unique_e_i_idx, tf.int32)
        outer_e_i_addends = tf.reshape(outer_e_i_addends, (-1,6))
        e_i_14_addends, e_i_42_addends, e_i_25_addends, e_i_53_addends, e_i_36_addends, e_i_61_addends = tf.unstack(outer_e_i_addends, axis=1)
        e_i_14s -= e_i_14_addends
        e_i_42s += e_i_42_addends
        e_i_25s -= e_i_25_addends
        e_i_53s += e_i_53_addends
        e_i_36s -= e_i_36_addends
        e_i_61s += e_i_61_addends

        bl_updates = tf.stack((e_i_14s, e_i_46s, e_i_61s), axis=1)
        t_updates = tf.stack((e_i_25s, e_i_45s, e_i_42s), axis=1)
        br_updates = tf.stack((e_i_56s, e_i_36s, e_i_53s), axis=1)
        m_updates = tf.stack((e_i_45s, e_i_56s, e_i_46s), axis=1)

        new_facetoedgemap = tf.scatter_nd(
            tf.expand_dims((nf_i_bls, nf_i_ts, nf_i_brs, nf_i_ms), axis=-1),
            (bl_updates, t_updates, br_updates, m_updates),
            new_facetoedgemap.shape)

        # reorder faces for correct normal orientation
        # new_edges, new_facetoedgemap, new_faces = np.array(new_edges), np.array(new_facetoedgemap), np.array(new_faces)
        centroid = tf.math.reduce_mean(new_vertices, axis=0)
        a_s = tf.gather(new_vertices, tf.gather(new_faces, 0, axis=1))
        b_s = tf.gather(new_vertices, tf.gather(new_faces, 1, axis=1)) 
        c_s = tf.gather(new_vertices, tf.gather(new_faces, 2, axis=1))
        normals = tf.linalg.cross(a_s - b_s, c_s - b_s)
        normal_magnitudes = tf_safe_norm(normals)
        # handle zero vectors
        normal_magnitudes += tf.cast(normal_magnitudes==0., normal_magnitudes.dtype)
        normals /= normal_magnitudes[:,None]
        face_centers = tf.math.reduce_mean(tf.gather(new_vertices, new_faces), axis=1)
        k_s = -tf.math.reduce_sum(normals*face_centers, axis=-1)
        centroid_signs = tf.math.sign(tf.math.reduce_sum(normals*centroid[None,:], axis=-1) + k_s)
        normal_signs = tf.math.sign(tf.math.reduce_sum(normals*(face_centers+normals), axis=-1) + k_s)
        new_faces = tf.tensor_scatter_nd_update(new_faces, tf.expand_dims(tf.where(normal_signs != centroid_signs), axis=-1), tf.reverse(tf.gather(new_faces, tf.where(normal_signs != centroid_signs)), [-1]))
        new_facetoedgemap = tf.tensor_scatter_nd_update(new_facetoedgemap, tf.where(normal_signs != centroid_signs),\
            tf.concat((tf.gather(new_facetoedgemap, tf.where(normal_signs != centroid_signs))[...,1], tf.gather(new_facetoedgemap, tf.where(normal_signs != centroid_signs))[...,0], tf.gather(new_facetoedgemap, tf.where(normal_signs != centroid_signs))[...,2]), axis=-1))
        # make new vertextofacemap
        vertex_indices = tf.range(new_vertices.shape[0], dtype=tf.int32)
        face_indices = tf.range(new_faces.shape[0], dtype=tf.int32)
        flattened_faces = tf.reshape(new_faces, (-1))
        new_vertextofacemap = vertex_indices[:,None] == flattened_faces[None,:]
        new_vertextofacemap = tf.cast(new_vertextofacemap, tf.int32)
        new_vertextofacemap = tf.reshape(new_vertextofacemap, (vertex_indices.shape[0], face_indices.shape[0], 3))
        new_vertextofacemap = tf.math.reduce_sum(new_vertextofacemap, axis=-1)
        new_vertextofacemap = tf.where(new_vertextofacemap!=0)
        new_vertextofacemap = tf.RaggedTensor.from_value_rowids(new_vertextofacemap[:,1], new_vertextofacemap[:,0])

        # make new facetoedgesign map
        nftes_col_1 = tf.gather(new_edges, new_facetoedgemap[:,0])[:,0] == new_faces[:,0]
        nftes_col_2 = tf.gather(new_edges, new_facetoedgemap[:,1])[:,0] == new_faces[:,1]
        nftes_col_3 = tf.gather(new_edges, new_facetoedgemap[:,2])[:,0] == new_faces[:,2]
        new_facetoedgesign = tf.stack((nftes_col_1, nftes_col_2, nftes_col_3), axis=1)

        if barycentric_faces is not None and bary_face_to_unsubbed_face is not None and barycentric_edges is not None:
            new_barycentric_faces = tf.zeros(tuple(new_faces.shape)+(2,), dtype=tf.float32)
            new_barycentric_edges = tf.zeros(tuple(new_faces.shape)+(2,2,), dtype=tf.float32)
            new_bary_face_to_unsubbed_face = tf.zeros(new_barycentric_faces.shape[0], dtype=bary_face_to_unsubbed_face.dtype)
            # bary vertices will be b300, b030, b003
            bary_v1s, bary_v2s, bary_v3s = tf.unstack(barycentric_faces, axis=-2)
            bary_e1s, bary_e2s, bary_e3s = tf.unstack(barycentric_edges, axis=-3)
            bary_v4s, bary_v5s, bary_v6s = tf.math.reduce_mean(bary_e1s, axis=-2), tf.math.reduce_mean(bary_e2s, axis=-2), tf.math.reduce_mean(bary_e3s, axis=-2)
            nf_i_bls_update = tf.stack((bary_v1s, bary_v4s, bary_v6s), axis=-2)
            new_barycentric_faces = tf.tensor_scatter_nd_update(new_barycentric_faces, tf.expand_dims(nf_i_bls, axis=-1), nf_i_bls_update)
            nf_i_bls_update = tf.stack((tf.stack((bary_v1s, bary_v4s), axis=-2), tf.stack((bary_v4s, bary_v6s), axis=-2), tf.stack((bary_v6s, bary_v1s), axis=-2)), axis=-3)
            new_barycentric_edges = tf.tensor_scatter_nd_update(new_barycentric_edges, tf.expand_dims(nf_i_bls, -1), nf_i_bls_update)
            nf_i_ts_update = tf.stack((bary_v2s, bary_v5s, bary_v4s), axis=-2)
            new_barycentric_faces = tf.tensor_scatter_nd_update(new_barycentric_faces, tf.expand_dims(nf_i_ts, axis=-1), nf_i_ts_update)
            nf_i_ts_update = tf.stack((tf.stack((bary_v2s, bary_v5s), axis=-2), tf.stack((bary_v4s, bary_v5s), axis=-2), tf.stack((bary_v4s, bary_v2s), axis=-2)), axis=-3)
            new_barycentric_edges = tf.tensor_scatter_nd_update(new_barycentric_edges, tf.expand_dims(nf_i_ts, -1), nf_i_ts_update)
            nf_i_brs_update = tf.stack((bary_v5s, bary_v6s, bary_v3s), axis=-2)
            new_barycentric_faces = tf.tensor_scatter_nd_update(new_barycentric_faces, tf.expand_dims(nf_i_brs, axis=-1), nf_i_brs_update)
            nf_i_brs_update = tf.stack((tf.stack((bary_v5s, bary_v6s), axis=-2), tf.stack((bary_v6s, bary_v3s), axis=-2), tf.stack((bary_v3s, bary_v5s), axis=-2)), axis=-3)
            new_barycentric_edges = tf.tensor_scatter_nd_update(new_barycentric_edges, tf.expand_dims(nf_i_brs, -1), nf_i_brs_update)
            nf_i_ms_update = tf.stack((bary_v4s, bary_v5s, bary_v6s), axis=-2)
            new_barycentric_faces = tf.tensor_scatter_nd_update(new_barycentric_faces, tf.expand_dims(nf_i_ms, axis=-1), nf_i_ms_update)
            nf_i_ms_update = tf.stack((tf.stack((bary_v4s, bary_v5s), axis=-2), tf.stack((bary_v5s, bary_v6s), axis=-2), tf.stack((bary_v4s, bary_v6s), axis=-2)), axis=-3)
            new_barycentric_edges = tf.tensor_scatter_nd_update(new_barycentric_edges, tf.expand_dims(nf_i_ms, -1), nf_i_ms_update)
            new_barycentric_faces = tf.tensor_scatter_nd_update(new_barycentric_faces, tf.expand_dims(tf.where(normal_signs != centroid_signs), axis=-1), tf.reverse(tf.gather(new_barycentric_faces, tf.where(normal_signs != centroid_signs)), [-2]))
            new_barycentric_edges = tf.tensor_scatter_nd_update(new_barycentric_edges, tf.where(normal_signs != centroid_signs),\
                tf.concat((tf.gather(new_barycentric_edges, tf.where(normal_signs != centroid_signs))[...,1,:,:], tf.gather(new_barycentric_edges, tf.where(normal_signs != centroid_signs))[...,0,:,:], tf.gather(new_barycentric_edges, tf.where(normal_signs != centroid_signs))[...,2,:,:]), axis=-3))
            new_bary_face_to_unsubbed_face = tf.tensor_scatter_nd_update(new_bary_face_to_unsubbed_face, tf.expand_dims(nf_i_bls, axis=-1), bary_face_to_unsubbed_face)
            new_bary_face_to_unsubbed_face = tf.tensor_scatter_nd_update(new_bary_face_to_unsubbed_face, tf.expand_dims(nf_i_ts, axis=-1), bary_face_to_unsubbed_face)
            new_bary_face_to_unsubbed_face = tf.tensor_scatter_nd_update(new_bary_face_to_unsubbed_face, tf.expand_dims(nf_i_brs, axis=-1), bary_face_to_unsubbed_face)
            new_bary_face_to_unsubbed_face = tf.tensor_scatter_nd_update(new_bary_face_to_unsubbed_face, tf.expand_dims(nf_i_ms, axis=-1), bary_face_to_unsubbed_face)

            return new_vertices, new_faces, new_edges, new_facetoedgemap, new_vertextofacemap, new_facetoedgesign, subdivisions+1, new_barycentric_faces, new_barycentric_edges, new_bary_face_to_unsubbed_face

    return new_vertices, new_faces, new_edges, new_facetoedgemap, new_vertextofacemap, new_facetoedgesign, subdivisions+1

def subdivide_tri_tf_paged(vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions):
    batch_size = 100
    main_len = len(vertices)//batch_size
    remainder = tf.math.floormod(len(vertices),batch_size)
    paged_vertices = tf.split(vertices, [batch_size]*(main_len)+[remainder])
    main_vertices = tf.stack(paged_vertices[:-1])
    remainder_vertices = tf.expand_dims(paged_vertices[-1], 0)

    def batched_subdivide(input_tuple):
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions, batch_size = input_tuple
        subdivision_counter = tf.constant(0)
        def not_done_dividing(vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivision_counter):
            return tf.math.less(subdivision_counter, subdivisions)
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivision_counter = tf.while_loop(
            not_done_dividing, subdivide_tri_tf, (vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivision_counter))
        predicted_surface_points = tf.stack(tf.split(vertices, batch_size.numpy()))
        return predicted_surface_points

    def get_batched_subdivide_input(vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions):
            batches = vertices.shape[0]
            batch_size = vertices.shape[-3]
            vertices_per_blob = vertices.shape[-2]
            faces_shape = faces.shape
            edges_shape = edges.shape
            batched_faces = faces[None,:,:] + tf.cast(vertices_per_blob*tf.range(batch_size), tf.int32)[:,None,None]
            batched_edges = edges[None,:,:] + tf.cast(vertices_per_blob*tf.range(batch_size), tf.int32)[:,None,None]
            batched_facetoedgemap = facetoedgemap[None,:,:] + tf.cast(edges_shape[0]*tf.range(batch_size), tf.int32)[:,None,None]
            batched_facetoedgesign = tf.broadcast_to(facetoedgesign[None,...], (batch_size,)+facetoedgesign.shape)
            batched_faces = tf.broadcast_to(batched_faces, (batches,) + batched_faces.shape)
            batched_edges = tf.broadcast_to(batched_edges, (batches,) + batched_edges.shape)
            batched_facetoedgemap = tf.broadcast_to(batched_facetoedgemap, (batches,) + batched_facetoedgemap.shape)
            batched_facetoedgesign = tf.broadcast_to(batched_facetoedgesign, (batches,) + batched_facetoedgesign.shape)
            vertices = tf.reshape(vertices, (batches, -1, vertices.shape[-1]))
            batched_faces = tf.reshape(batched_faces, (batches, -1, batched_faces.shape[-1]))
            batched_edges = tf.reshape(batched_edges, (batches, -1, batched_edges.shape[-1]))
            batched_facetoedgemap = tf.reshape(batched_facetoedgemap, (batches, -1, batched_facetoedgemap.shape[-1]))
            tiles = tf.constant((batches,1,1))
            batched_vertextofacemap = vertextofacemap[None,...] + tf.cast(faces_shape[0]*tf.range(batch_size), tf.int32)[:,None,None]
            batched_vertextofacemap = batched_vertextofacemap.merge_dims(0, 1)
            batched_vertextofacemap = tf.tile(batched_vertextofacemap[None,...], tiles)
            batched_facetoedgesign = tf.reshape(batched_facetoedgesign, (batches, -1, batched_facetoedgesign.shape[-1]))
            return (vertices, batched_faces, batched_edges, batched_facetoedgemap, batched_vertextofacemap, batched_facetoedgesign, tf.broadcast_to(subdivisions,(batches,)), tf.broadcast_to(batch_size, (batches,)))

    main_input = get_batched_subdivide_input(main_vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions)
    subdivided_main = tf.map_fn(batched_subdivide, main_input, dtype=tf.float32, fn_output_signature=tf.float32)
    subdivided_main = tf.reshape(subdivided_main, tf.concat(((-1,), subdivided_main.shape[-2:]), 0))

    if remainder == 0:
        return subdivided_main

    remainder_input = get_batched_subdivide_input(remainder_vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions)
    subdivided_remainder = tf.map_fn(batched_subdivide, remainder_input, dtype=tf.float32, fn_output_signature=tf.float32)
    subdivided_remainder = tf.reshape(subdivided_remainder, tf.concat(((-1,), subdivided_remainder.shape[-2:]), 0))
    return tf.concat((subdivided_main, subdivided_remainder), 0)

def subdivide_tri(vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, vertex_normals=None, sub_method="pn", just_compute_verts=False, refinement_level=1, control_points=None):
    # sub_method can be linear or point-normal
    num_orig_vertices = len(vertices)

    # compute new vertices on edges
    e2nv = np.zeros(len(edges), dtype=np.uint32)
    if sub_method == "linear":
        for edge_index, e in enumerate(edges):
            new_vertex = (vertices[e[0]] + vertices[e[1]]
                          ) / 2  # use mean for now
            vertices = np.append(vertices, [new_vertex], axis=0)  # yuck numpy
            # need to build an edge to new_vertex map to find it when building new tris
            e2nv[edge_index] = len(vertices)-1
    elif sub_method == "bicubic":
        assert(control_points is not None)
        half, zero = tf.constant(.5, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
        st = eval_triangles_tf(control_points, s=half, t=half)
        tu = eval_triangles_tf(control_points, s=zero, t=half)
        us = eval_triangles_tf(control_points, s=half, t=zero)
        for edge_index, e in enumerate(edges):
            for face_index, f in enumerate(facetoedgemap):
                if edge_index in f:
                    if f[0] == edge_index:
                        location = st
                    elif f[1] == edge_index:
                        location = tu
                    else:
                        location = us
                    # location = (st, tu, us)[np.argmax(f==edge_index)]
                    new_vertex = location[...,face_index,:].numpy()
                    vertices = np.append(vertices, [new_vertex], axis=0)
                    e2nv[edge_index] = len(vertices)-1
                    break
    else:  # point-normal triangles
        if vertex_normals is None:
            vertex_normals = get_vertex_normals(vertices, faces, vertextofacemap)

        # (1) compute new vertices, one for each edge, and add to vertex list
        edge_indices = np.arange(len(edges))
        first_edge_vertices, second_edge_vertices = vertices[edges[:,0]], vertices[edges[:,1]]
        first_edge_vertex_normals, second_edge_vertex_normals = vertex_normals[edges[:,0]], vertex_normals[edges[:,1]]
        new_vertices = point_normal_interpolate(first_edge_vertices, second_edge_vertices, first_edge_vertex_normals, second_edge_vertex_normals, 0.5)
        vertices = np.concatenate((vertices, new_vertices))
        e2nv += len(vertices) - 1 - np.arange(len(edges), dtype=np.uint32)[::-1]

    new_vertices = vertices
    new_faces = np.empty((len(faces)*4, 3), dtype=int)
    new_edges = np.empty((2*len(edges)+3*len(faces), 2), dtype=int)
    new_facetoedgemap = np.empty((len(new_faces), 3), dtype=int)
    E = len(edges)
    if not just_compute_verts:

        face_indices = np.arange(len(faces))
        face_edges = edges[facetoedgemap]
        v_1s, v_2s, v_3s = faces.T
        v_1_in_edge = np.sum(face_edges == v_1s[:,None,None], axis=-1)
        v_2_in_edge = np.sum(face_edges == v_2s[:,None,None], axis=-1)
        v_3_in_edge = np.sum(face_edges == v_3s[:,None,None], axis=-1)
        e_i_1s = facetoedgemap[np.logical_and(v_1_in_edge, v_2_in_edge)]
        e_i_2s = facetoedgemap[np.logical_and(v_2_in_edge, v_3_in_edge)]
        e_i_3s = facetoedgemap[np.logical_and(v_1_in_edge, v_3_in_edge)]

        v_4s, v_5s, v_6s = e2nv[e_i_1s], e2nv[e_i_2s], e2nv[e_i_3s]

        nf_i_bls = 4*face_indices
        nf_i_ts = 4*face_indices+1
        nf_i_brs = 4*face_indices+2
        nf_i_ms = 4*face_indices+3
        new_faces[nf_i_bls] = np.array((v_1s, v_6s, v_4s)).T
        new_faces[nf_i_ts] = np.array((v_2s, v_4s, v_5s)).T
        new_faces[nf_i_brs] = np.array((v_5s, v_6s, v_3s)).T
        new_faces[nf_i_ms] = np.array((v_4s, v_6s, v_5s)).T

        e_14s, e_42s, e_25s, e_53s, e_36s, e_61s, e_46s, e_45s, e_56s = np.sort((
            (v_1s, v_4s),
            (v_4s, v_2s),
            (v_2s, v_5s),
            (v_5s, v_3s),
            (v_3s, v_6s),
            (v_6s, v_1s),
            (v_4s, v_6s),
            (v_4s, v_5s),
            (v_5s, v_6s)), axis=1).transpose((0,2,1))

        e_i_14s = 2*e_i_1s
        e_i_42s = 2*e_i_1s+1
        e_i_14_addends = np.any(new_edges[e_i_14s] != e_14s, axis=-1)
        # e_i_14s += e_i_14_addends
        # e_i_42s -= e_i_14_addends
        e_i_25s = 2*e_i_2s
        e_i_53s = 2*e_i_2s+1
        e_i_25_addends = np.any(new_edges[e_i_25s] != e_25s, axis=-1)
        # e_i_25s += e_i_25_addends
        # e_i_53s -= e_i_25_addends
        e_i_36s = 2*e_i_3s
        e_i_61s = 2*e_i_3s+1
        e_i_36_addends = np.any(new_edges[e_i_36s] != e_36s, axis=-1)
        # e_i_36s += e_i_36_addends
        # e_i_61s -= e_i_36_addends
        e_i_46s = 2*E+3*face_indices
        e_i_45s = 2*E+3*face_indices+1
        e_i_56s = 2*E+3*face_indices+2

        # new_edges[e_i_14s] = e_14s
        # new_edges[e_i_42s] = e_42s
        # new_edges[e_i_25s] = e_25s
        # new_edges[e_i_53s] = e_53s
        # new_edges[e_i_36s] = e_36s
        # new_edges[e_i_61s] = e_61s
        # new_edges[e_i_46s] = e_46s
        # new_edges[e_i_45s] = e_45s
        # new_edges[e_i_56s] = e_56s
        
        # new_facetoedgemap[nf_i_bls] = np.array((e_i_14s, e_i_46s, e_i_61s)).T
        # new_facetoedgemap[nf_i_ts] = np.array((e_i_25s, e_i_45s, e_i_42s)).T
        # new_facetoedgemap[nf_i_brs] = np.array((e_i_56s, e_i_53s, e_i_36s)).T
        # new_facetoedgemap[nf_i_ms] = np.array((e_i_45s, e_i_56s, e_i_46s)).T
        for face_index in np.arange(len(faces)):

            nf_i_bl = nf_i_bls[face_index]
            nf_i_t = nf_i_ts[face_index]
            nf_i_br = nf_i_brs[face_index]
            nf_i_m = nf_i_ms[face_index]
            e_14 = e_14s[face_index]
            e_42 = e_42s[face_index]
            e_25 = e_25s[face_index]
            e_53 = e_53s[face_index]
            e_36 = e_36s[face_index]
            e_61 = e_61s[face_index]
            e_46 = e_46s[face_index]
            e_45 = e_45s[face_index]
            e_56 = e_56s[face_index]
            e_i_1 = e_i_1s[face_index]
            e_i_2 = e_i_2s[face_index]
            e_i_3 = e_i_3s[face_index]

            e_i_14 = 2*e_i_1
            e_i_42 = 2*e_i_1+1
            e_i_14_addend = np.any(new_edges[e_i_14] != e_14)
            e_i_14 += e_i_14_addend
            e_i_42 -= e_i_14_addend
            e_i_25 = 2*e_i_2
            e_i_53 = 2*e_i_2+1
            e_i_25_addend = np.any(new_edges[e_i_25] != e_25)
            e_i_25 += e_i_25_addend
            e_i_53 -= e_i_25_addend
            e_i_36 = 2*e_i_3
            e_i_61 = 2*e_i_3+1
            e_i_36_addend = np.any(new_edges[e_i_36] != e_36)
            e_i_36 += e_i_36_addend
            e_i_61 -= e_i_36_addend
            e_i_46 = 2*E+3*face_index
            e_i_45 = 2*E+3*face_index+1
            e_i_56 = 2*E+3*face_index+2

            new_edges[e_i_14] = e_14
            new_edges[e_i_42] = e_42
            new_edges[e_i_25] = e_25
            new_edges[e_i_53] = e_53
            new_edges[e_i_36] = e_36
            new_edges[e_i_61] = e_61
            new_edges[e_i_46] = e_46
            new_edges[e_i_45] = e_45
            new_edges[e_i_56] = e_56

            new_facetoedgemap[nf_i_bl] = e_i_14, e_i_46, e_i_61
            new_facetoedgemap[nf_i_t] = e_i_25, e_i_45, e_i_42
            new_facetoedgemap[nf_i_br] = e_i_56, e_i_53, e_i_36
            new_facetoedgemap[nf_i_m] = e_i_45, e_i_56, e_i_46

        # reorder faces for correct normal orientation
        # new_edges, new_facetoedgemap, new_faces = np.array(new_edges), np.array(new_facetoedgemap), np.array(new_faces)
        centroid = np.mean(new_vertices, axis=0)
        a_s, b_s, c_s = new_vertices[new_faces[:,0]], new_vertices[new_faces[:,1]], new_vertices[new_faces[:,2]]
        normals = np.cross(a_s - b_s, c_s - b_s)
        normals /= np.linalg.norm(normals, axis=-1)[:,None]
        face_centers = np.mean(new_vertices[new_faces], axis=1)
        k_s = -np.sum(normals*face_centers, axis=-1)
        centroid_signs = np.sign(np.sum(normals*centroid[None,:], axis=-1) + k_s)
        normal_signs = np.sign(np.sum(normals*(face_centers+normals), axis=-1) + k_s)
        new_faces[tuple(np.where(normal_signs != centroid_signs))] = np.flip(new_faces[tuple(np.where(normal_signs != centroid_signs))], axis=-1)
        new_facetoedgemap[tuple(np.where(normal_signs != centroid_signs))] = np.flip(new_facetoedgemap[tuple(np.where(normal_signs != centroid_signs))], axis=-1)
        assert(np.all(new_faces == new_faces))

        # make new vertextofacemap
        new_vertextofacemap = [[] for v in range(len(new_vertices))]
        for face_index in range(len(new_faces)):
            face = new_faces[face_index]
            for vertex in face:
                new_vertextofacemap[vertex].append(face_index)

        # make new facetoedgesign map
        new_facetoedgesign = []
        for f_idx, f in enumerate(new_faces):
            new_facetoedgesign.append([new_edges[new_facetoedgemap[f_idx][0]][0] == f[0],
                                       new_edges[new_facetoedgemap[f_idx]
                                                 [1]][0] == f[1],
                                       new_edges[new_facetoedgemap[f_idx][2]][0] == f[2]])

    return new_vertices, new_faces, new_edges, new_facetoedgemap, new_vertextofacemap, new_facetoedgesign

def icosahedron():
    """
    Build vertices, faces, and edges for an icosahedron
    vertices is a (12,3) array of floats
    faces is a (20,3) array of ints that are indices into vertices
    edges is a (30,2) array of intes that are indices into vertices
    facetoedgemap is a (20,3) array of ints that are indices into edges
    vertextofacemap is a (12,5) array of ints that are indices in faces
    """

    # spherical coordinates generation of vertex positions (icosa oriented as if balanced on a point)
    lati0 = np.deg2rad(90)
    lati1 = np.arctan(1/2)
    lati2 = -np.arctan(1/2)
    lati3 = -np.deg2rad(90)

    lon1 = 0
    lon2 = np.deg2rad(72*1)
    lon3 = np.deg2rad(72*2)
    lon4 = np.deg2rad(72*3)
    lon5 = np.deg2rad(72*4)

    lon6 = np.deg2rad(36)
    lon7 = np.deg2rad(36+72*1)
    lon8 = np.deg2rad(36+72*2)
    lon9 = np.deg2rad(36+72*3)
    lon10 = np.deg2rad(36+72*4)

    # vertex positions
    # cupy doesn't like 0 entries in this array, so use sin(0) instead
    vertices = np.array([[np.sin(0), np.sin(0), np.sin(lati0)],
                         [np.cos(lati1)*np.cos(lon1), np.cos(lati1)
                          * np.sin(lon1), np.sin(lati1)],
                         [np.cos(lati1)*np.cos(lon2), np.cos(lati1)
                          * np.sin(lon2), np.sin(lati1)],
                         [np.cos(lati1)*np.cos(lon3), np.cos(lati1)
                          * np.sin(lon3), np.sin(lati1)],
                         [np.cos(lati1)*np.cos(lon4), np.cos(lati1)
                          * np.sin(lon4), np.sin(lati1)],
                         [np.cos(lati1)*np.cos(lon5), np.cos(lati1)
                          * np.sin(lon5), np.sin(lati1)],
                         [np.cos(lati2)*np.cos(lon6), np.cos(lati2)
                          * np.sin(lon6), np.sin(lati2)],
                         [np.cos(lati2)*np.cos(lon7), np.cos(lati2)
                          * np.sin(lon7), np.sin(lati2)],
                         [np.cos(lati2)*np.cos(lon8), np.cos(lati2)
                          * np.sin(lon8), np.sin(lati2)],
                         [np.cos(lati2)*np.cos(lon9), np.cos(lati2)
                          * np.sin(lon9), np.sin(lati2)],
                         [np.cos(lati2)*np.cos(lon10), np.cos(lati2)
                          * np.sin(lon10), np.sin(lati2)],
                         [np.sin(0), np.sin(0), np.sin(lati3)]
                         ])

    # face indices (counterclockwise when looking from the outside)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
                      [2, 1, 6], [3, 2, 7], [4, 3, 8], [5, 4, 9], [1, 5, 10],
                      [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1],
                      [11, 7, 6], [11, 8, 7], [11, 9, 8], [
                          11, 10, 9], [11, 6, 10]
                      ],
                     dtype=np.uint32)

    # edge indices
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0],
                      [3, 4], [4, 0], [4, 5], [5, 0], [5, 1],
                      [1, 6], [6, 2], [2, 7], [7, 3],
                      [3, 8], [8, 4], [4, 9], [9, 5],
                      [5, 10], [10, 1], [6, 7], [7, 8],
                      [8, 9], [9, 10], [10, 6], [11, 7], [6, 11],
                      [11, 8], [11, 9], [11, 10]
                      ],
                     dtype=np.uint32)

    # edge indices for each face, could be easily calculated
    facetoedgemap = np.array(
        [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0],
         [1, 10, 11], [3, 12, 13], [5, 14, 15], [7, 16, 17], [9, 18, 19],
         [20, 12, 11], [21, 14, 13], [22, 16, 15], [23, 18, 17], [24, 10, 19],
         [25, 20, 26], [27, 21, 25], [28, 22, 27], [29, 23, 28], [26, 24, 29]
         ],
        dtype=np.uint32)

    # this screws up the faces for some reason
    # facetoedgemap = []
    # for f in faces:
    #     # find all edges that are in a face
    #     accum = []
    #     for idx, e in enumerate(edges):
    #         if (f[0] in e) and (f[1] in e):
    #             accum.append(idx)
    #         if (f[1] in e) and (f[2] in e):
    #             accum.append(idx)
    #         if (f[2] in e) and (f[0] in e):
    #             accum.append(idx)
    #     facetoedgemap.append(accum)

    # this could easily be auto-calculated. in general this is a list, not a 2D array
    vertextofacemap = [[0, 1, 2, 3, 4], [0, 4, 5, 9, 14], [0, 1, 5, 6, 10],
                       [1, 2, 6, 7, 11], [2, 3, 7, 8, 12], [3, 4, 8, 9, 13],
                       [5, 10, 14, 15, 19], [6, 10, 11, 15, 16], [
                           7, 11, 12, 16, 17],
                       [8, 12, 13, 17, 18], [9, 13, 14, 18, 19], [15, 16, 17, 18, 19]]

    # make new facetoedgesign map
    facetoedgesign = []
    for f_idx, f in enumerate(faces):
        facetoedgesign.append([edges[facetoedgemap[f_idx][0]][0] == f[0],
                              edges[facetoedgemap[f_idx][1]][0] == f[1], edges[facetoedgemap[f_idx][2]][0] == f[2]])

    return vertices[:,[2,1,0]], faces, edges, facetoedgemap, vertextofacemap, facetoedgesign

def triangle_areas(triangles):
    As = triangles[:,1] - triangles[:,0]
    Bs = triangles[:,2] - triangles[:,0]
    return np.linalg.norm(np.cross(As,Bs), axis=1) / 2

def tetrahedron_volumes(triangles, center):
    As = triangles[:,0] - center
    Bs = triangles[:,1] - center
    Cs = triangles[:,2] - center
    return np.abs(np.einsum('ij,ij->i', np.cross(As,Bs), Cs) / 6)

def get_planar_face_equations(vertices, faces):
    face_vertices = vertices[faces]
    es = np.cross(face_vertices[:, 0,:] - face_vertices[:, 2,:], face_vertices[:, 1,:] - face_vertices[:, 2,:])
    ks = -np.einsum("ij,ij->i", es, face_vertices[:, 0,:])
    return es, ks

def inside_tetrahedrons(vertices, faces, center, points):
    points = tf.constant(points, dtype=tf.float32)
    face_vertices = tf.constant(vertices[faces], dtype=tf.float32)
    center = tf.constant(center, dtype=tf.float32)
    es1 = tf.linalg.cross(face_vertices[:, 0,:] - face_vertices[:, 2,:], face_vertices[:, 1,:] - face_vertices[:, 2,:])
    ks1 = -tf.einsum("ij,ij->i", es1, face_vertices[:, 0,:])
    es2 = tf.linalg.cross(face_vertices[:, 0,:] - center[None,:], face_vertices[:, 1,:] - center[None,:])
    ks2 = -tf.einsum("ij,ij->i", es2, face_vertices[:, 0,:])
    es3 = tf.linalg.cross(face_vertices[:, 0,:] - center[None,:], face_vertices[:, 2,:] - center[None,:])
    ks3 = -tf.einsum("ij,ij->i", es3, face_vertices[:, 0,:])
    es4 = tf.linalg.cross(face_vertices[:, 1,:] - center[None,:], face_vertices[:, 2,:] - center[None,:])
    ks4 = -tf.einsum("ij,ij->i", es4, face_vertices[:, 1,:])
    interior_points = tf.math.reduce_mean(tf.concat((face_vertices,tf.broadcast_to(center,(face_vertices.shape[0],1,3))), axis=1), axis=1)
    inside_bound_1_signs = get_tetrahedron_signs((es1,ks1), interior_points)
    inside_bound_1_signs = tf.broadcast_to(inside_bound_1_signs[:,None], (len(inside_bound_1_signs), len(points)))
    inside_bound_2_signs = get_tetrahedron_signs((es2,ks2), interior_points)
    inside_bound_2_signs = tf.broadcast_to(inside_bound_2_signs[:,None], (len(inside_bound_2_signs), len(points)))
    inside_bound_3_signs = get_tetrahedron_signs((es3,ks3), interior_points)
    inside_bound_3_signs = tf.broadcast_to(inside_bound_3_signs[:,None], (len(inside_bound_3_signs), len(points)))
    inside_bound_4_signs = get_tetrahedron_signs((es4,ks4), interior_points)
    inside_bound_4_signs = tf.broadcast_to(inside_bound_4_signs[:,None], (len(inside_bound_4_signs), len(points)))
    bound_1_signs = get_planar_face_signs((es1,ks1), points)
    bound_2_signs = get_planar_face_signs((es2,ks2), points)
    bound_3_signs = get_planar_face_signs((es3,ks3), points)
    bound_4_signs = get_planar_face_signs((es4,ks4), points)
    inside_bound_1 = tf.math.logical_or(bound_1_signs==inside_bound_1_signs, bound_1_signs==0)
    inside_bound_2 = tf.math.logical_or(bound_2_signs==inside_bound_2_signs, bound_2_signs==0)
    inside_bound_3 = tf.math.logical_or(bound_3_signs==inside_bound_3_signs, bound_3_signs==0)
    inside_bound_4 = tf.math.logical_or(bound_4_signs==inside_bound_4_signs, bound_4_signs==0)
    inside = tf.math.reduce_any(inside_bound_1 & inside_bound_2 & inside_bound_3 & inside_bound_4, axis=0)
    return inside.numpy()

def get_tetrahedron_signs(tetrahedron_equations, ps):
    es, ks = tetrahedron_equations
    return tf.math.sign(tf.einsum("ij,ij->i", es, ps) + ks)

def get_planar_face_signs(planar_face_equations, ps):
    es, ks = planar_face_equations
    e_ps = np.einsum("ij,kj->ik", es, ps)
    e_ps += ks[:,None]
    return np.sign(e_ps)

# this is a vectorized approach which computes bounding prisms using each planar face's edges and a center point.
# for each face, equations for three bounding planes (in terms of a normal e_i and an offset k_i) are computed.
# then, the input point is plugged into these equations, and the signs of the resulting output are compared to signs
# produced when using a known interior point as input. since points at the intersection of two or more planes may
# will produce output near zero for those planes' equations, and since signs of outputs near zero may be imprecise,
# an evaluation near zero is considered a sign match. the face producing the most sign matches is considered the
# bounding face. if no bounding face is identified (float imprecision? or some other pathology?), return None.
def bounding_face(p, faces, vertices, center):
    face_vertices = vertices[faces]
    es = np.cross(np.concatenate((face_vertices[:, 0,:] - center, face_vertices[:, 0,:] - center, face_vertices[:, 1,:] - center)), np.concatenate((face_vertices[:, 1,:] - center, face_vertices[:, 2,:] - center, face_vertices[:, 2,:] - center)))
    ks = -np.einsum("ij,ij->i", es, np.concatenate((face_vertices[:, 0,:], face_vertices[:, 0,:], face_vertices[:, 1,:])))
    interior_points = np.mean(face_vertices, axis=1)
    interior_signs = np.sign(np.einsum("ij,ij->i", es, np.concatenate((interior_points, interior_points, interior_points))) + ks)
    e_ps = np.einsum("ij,j->i", es, p) + ks
    sign_matches = np.sign(e_ps) == interior_signs
    e_p_zeros = np.abs(e_ps) <= 0.0000001
    bounding = np.zeros((len(faces)), dtype=int)
    bounding += sign_matches[:len(faces)]
    bounding += sign_matches[len(faces):2*len(faces)]
    bounding += sign_matches[2*len(faces):]
    bounding += e_p_zeros[:len(faces)]
    bounding += e_p_zeros[len(faces):2*len(faces)]
    bounding += e_p_zeros[2*len(faces):]
    if np.max(bounding) >= 3:
        return np.argmax(bounding)
    return None

def bounding_faces(ps, faces, vertices, center):
    # ps: nx3
    # faces: mx3
    # vertices: lx3
    # face vertices: mx3x3
    # es: (m*3)x3
    # ks: (m*3)
    # interior points: (m*3)x3
    # interior signs: (m*3)
    # e_ps: (m*3)xn
    # sign_matches: (m*3)xn
    # e_p_zeros: (m*3)xn
    # bounding:  mxn
    face_vertices = vertices[faces]
    es = np.cross(np.concatenate((face_vertices[:, 0,:] - center, face_vertices[:, 0,:] - center, face_vertices[:, 1,:] - center)), np.concatenate((face_vertices[:, 1,:] - center, face_vertices[:, 2,:] - center, face_vertices[:, 2,:] - center)))
    ks = -np.einsum("ij,ij->i", es, np.concatenate((face_vertices[:, 0,:], face_vertices[:, 0,:], face_vertices[:, 1,:])))
    face_means = np.mean(face_vertices, axis=1)
    interior_points = np.broadcast_to(face_means, (3, face_means.shape[0], face_means.shape[1])).reshape((-1,3))
    interior_signs = np.sign(np.einsum("ij,ij->i", es, interior_points) + ks)
    interior_signs = np.broadcast_to(interior_signs, (len(ps), len(interior_signs))).T
    e_ps = np.einsum("ij,kj->ik", es, ps)
    e_ps += ks[:,None]
    sign_matches = np.sign(e_ps) == interior_signs
    e_p_zeros = np.abs(e_ps) <= 0.0000001
    bounding = np.zeros((len(faces), len(ps)), dtype=int)
    bounding += sign_matches[:len(faces)]
    bounding += sign_matches[len(faces):2*len(faces)]
    bounding += sign_matches[2*len(faces):]
    bounding += e_p_zeros[:len(faces)]
    bounding += e_p_zeros[len(faces):2*len(faces)]
    bounding += e_p_zeros[2*len(faces):]
    return np.argmax(bounding, axis=0)

def face_area_less_than(vertices, comparison):
    b300, b030, b003 = vertices
    s1 = b300 - b003
    s2 = b030 - b003
    s3 = b300 - b030
    # from area = (1/2)bh, triangle area must be at least (minimum_side_length**2)/1.25
    if np.min(np.sum((s1, s2, s3), axis=1)**2) >= comparison * 1.25:
        return False
    return np.linalg.norm(np.cross(b300 - b003, b030 - b003))/2 < comparison

def inside_icosahedron(p, vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, center=None, planar_face_equations=None, interior_planar_face_signs=None, vertex_normals=None):
    if center is None:
        center = np.mean(vertices, axis=0)
    if planar_face_equations is None:
        planar_face_equations = get_planar_face_equations(vertices, faces)
    if interior_planar_face_signs is None:
        interior_planar_face_signs = get_planar_face_signs(planar_face_equations, center)
    if np.all(get_planar_face_signs(planar_face_equations, p) == interior_planar_face_signs):
        return True
    face = bounding_face(p, faces, vertices, center)
    if face is None:
        return False
    face_area_small_enough = face_area_less_than(vertices[faces[face]], 0.5)
    while not face_area_small_enough:
        vertices = vertices[faces[face]]
        if vertex_normals is None:
            vertex_normals = get_vertex_normals(vertices, faces, vertextofacemap)
        faces = np.array([[0,1,2]])
        edges = np.array([[0,1],[1,2],[2,0]])
        facetoedgemap = np.array([[0,1,2]])
        vertextofacemap = [[0], [0], [0]]
        facetoedgesign = [facetoedgesign[face]]
        intersect = np.mean(vertices, axis=0)
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = subdivide_tri(
            vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, vertex_normals=vertex_normals, sub_method="pn")
        face = bounding_face(p, faces, vertices, center)
        if face is None:
            return np.linalg.norm(intersect - center) >= np.linalg.norm(p - center)
        face_area_small_enough = face_area_less_than(vertices[faces[face]], 0.5)
    vertices = vertices[faces[face]]
    faces = np.array([[0,1,2]])
    edges = np.array([[0,1],[1,2],[2,0]])
    facetoedgemap = [[0,1,2]]
    vertextofacemap = [[0], [0], [0]]
    facetoedgesign = [facetoedgesign[face]]
    intersect = np.mean(vertices, axis=0)
    return np.linalg.norm(intersect - center) >= np.linalg.norm(p - center)

def get_maximum_planar_face_area(vertices, faces):
    b300s = vertices[faces[:,0]]
    b030s = vertices[faces[:,1]]
    b003s = vertices[faces[:,2]]
    areas = np.linalg.norm(np.cross(b300s - b003s, b030s - b003s), axis=1)/2
    return np.max(areas)

def testprint():
    for _ in range(100):
        print("hello")

# from https://en.wikipedia.org/wiki/Point-normal_triangle
def omega_ij(i, j, vertices, normals):
    return(np.einsum('ij,ij->i', vertices[:,j-1,:] - vertices[:,i-1,:], normals[:,i-1,:]))
def get_control_point(i, j, vertices, normals):
    return((1/3) * ((2 * vertices[:,i-1,:]) + vertices[:,j-1,:] - (omega_ij(i, j, vertices, normals)[:,None] * normals[:,i-1,:])))
def get_control_points(vertices, normals):
    b300s, b030s, b003s = vertices[:,0,:], vertices[:,1,:], vertices[:,2,:]
    b012s = get_control_point(3, 2, vertices, normals)
    b021s = get_control_point(2, 3, vertices, normals)
    b102s = get_control_point(3, 1, vertices, normals)
    b201s = get_control_point(1, 3, vertices, normals)
    b120s = get_control_point(2, 1, vertices, normals)
    b210s = get_control_point(1, 2, vertices, normals)
    Es = (1/6) * (b012s + b021s + b102s + b201s + b120s + b210s)
    Vs = (1/3) * (b300s + b030s + b003s)
    b111s = Es + (1/2) * (Es - Vs)
    return np.stack((b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s), axis=1)

def dists_to_controls(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap, normals=None):
    center = np.array((0.,0.,0.))
    vertices = zyx_dists[:,None]*np.copy(zyx_vertices).reshape((-1,3))[...,::-1]
    faces = np.copy(zyx_faces).reshape((-1,3))
    triangles = vertices[faces]
    if normals is None:
        normals = get_vertex_normals(vertices, faces, vertextofacemap)
        normals = normals[faces]
    all_control_points = get_control_points(triangles, normals)
    return all_control_points[...,::-1]

def omega_ij_tf(i, j, vertices, normals):
    return(tf.einsum("ij,ij->i", vertices[...,j-1,:] - vertices[...,i-1,:], normals[...,i-1,:]))
def get_control_point_tf(i, j, vertices, normals):
    return((1/3) * ((2 * vertices[...,i-1,:]) + vertices[...,j-1,:] - (omega_ij_tf(i, j, vertices, normals)[...,None] * normals[...,i-1,:])))
def get_control_points_tf(vertices, normals):
    b300s, b030s, b003s = vertices[...,0,:], vertices[...,1,:], vertices[...,2,:]
    b012s = get_control_point_tf(3, 2, vertices, normals)
    b021s = get_control_point_tf(2, 3, vertices, normals)
    b102s = get_control_point_tf(3, 1, vertices, normals)
    b201s = get_control_point_tf(1, 3, vertices, normals)
    b120s = get_control_point_tf(2, 1, vertices, normals)
    b210s = get_control_point_tf(1, 2, vertices, normals)
    Es = (1/6) * (b012s + b021s + b102s + b201s + b120s + b210s)
    Vs = (1/3) * (b300s + b030s + b003s)
    b111s = Es + (1/2) * (Es - Vs)
    return tf.stack((b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s), axis=-2)
def get_get_control_points_tf_input(vertices, faces, vertextofacemap):
        if tf.rank(vertices) > 2:
            num_blobs = tf.math.reduce_prod(vertices.shape[:-2])
            new_vertices = tf.reshape(vertices, (-1, 3))
            vertices_per_blob = vertices.shape[-2]
            new_faces = faces[None,:,:] + vertices_per_blob*tf.range(num_blobs)[:,None,None]
            new_faces = tf.reshape(new_faces, (-1, 3))
            new_vertextofacemap = vertextofacemap[None,...] + tf.cast(faces.shape[0]*tf.range(num_blobs), tf.int32)[:,None,None]
            new_vertextofacemap = new_vertextofacemap.merge_dims(0, 1)
            return new_vertices, new_faces, new_vertextofacemap
        return vertices, faces, vertextofacemap
def dists_to_controls_tf(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap, normals=None):
    vertices = zyx_dists[...,None]*zyx_vertices
    new_vertices, new_faces, new_vertextofacemap = get_get_control_points_tf_input(vertices, zyx_faces, vertextofacemap)
    triangles = tf.gather(new_vertices, new_faces)
    if normals is None:
        normals = get_vertex_normals_tf(new_vertices, new_faces, new_vertextofacemap)
        normals = tf.gather(normals, new_faces)
    all_control_points = get_control_points_tf(triangles, normals)
    all_control_points = tf.reshape(all_control_points, tuple(vertices.shape[:-2]) + (len(zyx_faces), 10, 3))
    return all_control_points
# return points at barycentric centers of triangles (later should support arbitrary input coords)

@tf.function(input_signature=(tf.TensorSpec(shape=[None,10,3], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32)))
def eval_triangles_tf(all_control_points, s=None, t=None):
    if s is None or t is None:
        s, t = 1/3, 1/3
    u = 1-s-t
    b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s = tf.unstack(all_control_points, axis=-2)
    s, t, u = tf.expand_dims(s,-1), tf.expand_dims(t,-1), tf.expand_dims(u,-1)
    return (s**3)*b300s + (t**3)*b030s + (u**3)*b003s + 3*(s**2)*t*b210s + 3*s*(t**2)*b120s + \
        3*(t**2)*u*b021s + 3*t*(u**2)*b012s + 3*s*(u**2)*b102s + 3*(s**2)*u*b201s + 6*s*t*u*b111s

@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,10,3], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32)))
def eval_triangles_tf_batch(all_control_points, s=None, t=None):
    if s is None or t is None:
        s, t = 1/3, 1/3
    u = 1-s-t
    b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s = tf.unstack(all_control_points, axis=-2)
    s, t, u = tf.expand_dims(s,-1), tf.expand_dims(t,-1), tf.expand_dims(u,-1)
    s, t, u = tf.expand_dims(s,0), tf.expand_dims(t,0), tf.expand_dims(u,0)
    return (s**3)*b300s + (t**3)*b030s + (u**3)*b003s + 3*(s**2)*t*b210s + 3*s*(t**2)*b120s + \
        3*(t**2)*u*b021s + 3*t*(u**2)*b012s + 3*s*(u**2)*b102s + 3*(s**2)*u*b201s + 6*s*t*u*b111s

def dists_to_volume(zyx_start, zyx_vertices, zyx_faces, zyx_dists, edges, facetoedgemap, vertextofacemap, facetoedgesign):
    center = zyx_start[::-1]
    vertices = center + zyx_dists[:,None]*np.copy(zyx_vertices).reshape((-1,3))[:,[2,1,0]]
    faces = np.copy(zyx_faces).reshape((-1,3))
    max_area = get_maximum_planar_face_area(vertices, faces)
    while max_area > 0.75:
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = subdivide_tri(
            vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, sub_method="pn")
        max_area = get_maximum_planar_face_area(vertices, faces)
    triangles = vertices[faces]
    return np.sum(tetrahedron_volumes(triangles, center))

def icosahedron_bbox(zyx_start, zyx_vertices, zyx_faces, zyx_dists, zyx_shape, vertextofacemap):
    vertices = zyx_vertices[:,[2,1,0]]
    zyx_controls = dists_to_controls(vertices, zyx_faces, zyx_dists+1, vertextofacemap)
    zyx_controls += zyx_start
    zyx_controls = zyx_controls.reshape(-1,3)
    zyx_floors = np.min(np.floor(zyx_controls), axis=0).astype(int).clip(0, np.array(zyx_shape)-1)
    zyx_ceils = np.max(np.ceil(zyx_controls) + 1, axis=0).astype(int).clip(0, np.array(zyx_shape)-1)
    return zyx_floors[0], zyx_ceils[0], zyx_floors[1], zyx_ceils[1], zyx_floors[2], zyx_ceils[2]

def bounding_radius_outer(zyx_start, zyx_vertices, zyx_faces, zyx_dists, vertextofacemap):
    zyx_controls = dists_to_controls(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap)
    return np.max(np.sqrt(np.sum((zyx_controls-zyx_start)**2, axis=1)))

def bounding_radius_inner(zyx_start, zyx_vertices, zyx_faces, zyx_dists, vertextofacemap):
    zyx_controls = dists_to_controls(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap)
    return np.min(np.sqrt(np.sum((zyx_controls-zyx_start)**2, axis=1)))

def bounding_radius_outer_isotropic(zyx_vertices, zyx_faces, zyx_dists, anisotropy, vertextofacemap):
    zyx_center = np.array(zyx_dists[12:]) * np.array(anisotropy)
    zyx_controls = dists_to_controls(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap) * np.array(anisotropy)
    return np.max(np.sqrt(np.sum((zyx_controls-zyx_center)**2, axis=1)))

def bounding_radius_inner_isotropic(zyx_vertices, zyx_faces, zyx_dists, anisotropy, vertextofacemap):
    zyx_center = np.array(zyx_dists[12:]) * np.array(anisotropy)
    zyx_controls = dists_to_controls(zyx_vertices, zyx_faces, zyx_dists, vertextofacemap) * np.array(anisotropy)
    return np.min(np.sqrt(np.sum((zyx_controls-zyx_center)**2, axis=1)))

def pred_instance_to_control_points(cartesian_vertex_directions, cartesian_vertex_dists, other_control_dists, b111_barys, rays):
    first_cartesian_edge_points = tf.gather(cartesian_vertex_directions, rays.edges_tf[...,0])
    second_cartesian_edge_points = tf.gather(cartesian_vertex_directions, rays.edges_tf[...,1])
    cartesian_first_edge_control_directions = (2*first_cartesian_edge_points + second_cartesian_edge_points) / 3
    cartesian_second_edge_control_directions = (first_cartesian_edge_points + 2*second_cartesian_edge_points) / 3
    cartesian_first_edge_control_directions /= tf_safe_norm(cartesian_first_edge_control_directions)[...,None]
    cartesian_second_edge_control_directions /= tf_safe_norm(cartesian_second_edge_control_directions)[...,None]
    cartesian_first_edge_controls = cartesian_first_edge_control_directions * other_control_dists[...,:len(rays.edges_tf),None]
    cartesian_second_edge_controls = cartesian_second_edge_control_directions * other_control_dists[...,len(rays.edges_tf):2*len(rays.edges_tf),None]

    first_edge_indices = rays.facetoedgemap_tf[...,0]
    second_edge_indices = rays.facetoedgemap_tf[...,1]
    third_edge_indices = rays.facetoedgemap_tf[...,2]

    cartesian_vertices = cartesian_vertex_directions*cartesian_vertex_dists[...,None]
    b300s = tf.gather(cartesian_vertices, rays.faces_tf[...,0])
    b030s = tf.gather(cartesian_vertices, rays.faces_tf[...,1])
    b003s = tf.gather(cartesian_vertices, rays.faces_tf[...,2])

    b210s = tf.gather(cartesian_first_edge_controls, first_edge_indices)
    b120s = tf.gather(cartesian_second_edge_controls, first_edge_indices)
    b021s = tf.gather(cartesian_first_edge_controls, second_edge_indices)
    b012s = tf.gather(cartesian_second_edge_controls, second_edge_indices)
    b102s = tf.gather(cartesian_first_edge_controls, third_edge_indices)
    b201s = tf.gather(cartesian_second_edge_controls, third_edge_indices)
    swapped_e1_face_is = tf.where(~rays.facetoedgesign_tf[...,0])
    swapped_e2_face_is = tf.where(~rays.facetoedgesign_tf[...,1])
    swapped_e3_face_is = tf.where(~rays.facetoedgesign_tf[...,2])
    swapped_e1_is = tf.gather(first_edge_indices, swapped_e1_face_is)
    swapped_e2_is = tf.gather(second_edge_indices, swapped_e2_face_is)
    swapped_e3_is = tf.gather(third_edge_indices, swapped_e3_face_is)
    b210s = tf.tensor_scatter_nd_update(b210s, tf.expand_dims(swapped_e1_face_is, axis=-1), tf.gather(cartesian_second_edge_controls, swapped_e1_is))
    b120s = tf.tensor_scatter_nd_update(b120s, tf.expand_dims(swapped_e1_face_is, axis=-1), tf.gather(cartesian_first_edge_controls, swapped_e1_is))
    b021s = tf.tensor_scatter_nd_update(b021s, tf.expand_dims(swapped_e2_face_is, axis=-1), tf.gather(cartesian_second_edge_controls, swapped_e2_is))
    b012s = tf.tensor_scatter_nd_update(b012s, tf.expand_dims(swapped_e2_face_is, axis=-1), tf.gather(cartesian_first_edge_controls, swapped_e2_is))
    b102s = tf.tensor_scatter_nd_update(b102s, tf.expand_dims(swapped_e3_face_is, axis=-1), tf.gather(cartesian_second_edge_controls, swapped_e3_is))
    b201s = tf.tensor_scatter_nd_update(b201s, tf.expand_dims(swapped_e3_face_is, axis=-1), tf.gather(cartesian_first_edge_controls, swapped_e3_is))

    cart_v1_dirs = tf.gather(cartesian_vertex_directions, rays.faces_tf[...,0])
    cart_v2_dirs = tf.gather(cartesian_vertex_directions, rays.faces_tf[...,1])
    cart_v3_dirs = tf.gather(cartesian_vertex_directions, rays.faces_tf[...,2])
    if b111_barys is not None:
        b111_uv, b111_u_proportion = tf.split(b111_barys, num_or_size_splits=2, axis=-1)
        b111_u = b111_uv * b111_u_proportion
        b111_v = b111_uv - b111_u
        cartesian_b111_direction = b111_u[...,None]*cart_v1_dirs + b111_v[...,None]*cart_v2_dirs + (1-b111_u-b111_v)[...,None]*cart_v3_dirs
        cartesian_b111_direction /= tf_safe_norm(cartesian_b111_direction)[...,None]
    else:
        cartesian_b111_direction = (cart_v1_dirs + cart_v2_dirs + cart_v3_dirs) / 3
        cartesian_b111_direction /= tf_safe_norm_three(cartesian_b111_direction)[...,None]
    b111s = cartesian_b111_direction * other_control_dists[...,2*len(rays.edges_tf):,None]
    all_controls = tf.stack((b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s), axis=-2)
    return all_controls.numpy()

@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,3], dtype=tf.float32), tf.TensorSpec(shape=[None,None], dtype=tf.float32), tf.TensorSpec(shape=[None,None], dtype=tf.float32), tf.TensorSpec(shape=[None,2], dtype=tf.float32), tf.TensorSpec(shape=[None,2], dtype=tf.int32), tf.TensorSpec(shape=[None,3], dtype=tf.int32), tf.TensorSpec(shape=[None,3], dtype=tf.int32), tf.TensorSpec(shape=[None,3], dtype=tf.bool)))
def pred_instances_to_control_points(cartesian_vertex_directions, cartesian_vertex_dists, other_control_dists, b111_barys, edges_tf, faces_tf, facetoedgemap_tf, facetoedgesign_tf):
    first_cartesian_edge_points = tf.gather(cartesian_vertex_directions, edges_tf[...,0], axis=1)
    second_cartesian_edge_points = tf.gather(cartesian_vertex_directions, edges_tf[...,1], axis=1)
    cartesian_first_edge_control_directions = (2*first_cartesian_edge_points + second_cartesian_edge_points) / 3
    cartesian_second_edge_control_directions = (first_cartesian_edge_points + 2*second_cartesian_edge_points) / 3
    cartesian_first_edge_control_directions /= tf_safe_norm_three(cartesian_first_edge_control_directions)[...,None]
    cartesian_second_edge_control_directions /= tf_safe_norm_three(cartesian_second_edge_control_directions)[...,None]
    cartesian_first_edge_controls = cartesian_first_edge_control_directions * other_control_dists[...,:tf.shape(edges_tf)[0],None]
    cartesian_second_edge_controls = cartesian_second_edge_control_directions * other_control_dists[...,tf.shape(edges_tf)[0]:2*tf.shape(edges_tf)[0],None]

    first_edge_indices = facetoedgemap_tf[...,0]
    second_edge_indices = facetoedgemap_tf[...,1]
    third_edge_indices = facetoedgemap_tf[...,2]

    cartesian_vertices = cartesian_vertex_directions*cartesian_vertex_dists[...,None]
    b300s = tf.gather(cartesian_vertices, faces_tf[...,0], axis=1)
    b030s = tf.gather(cartesian_vertices, faces_tf[...,1], axis=1)
    b003s = tf.gather(cartesian_vertices, faces_tf[...,2], axis=1)

    b210s = tf.gather(cartesian_first_edge_controls, first_edge_indices, axis=1)
    b120s = tf.gather(cartesian_second_edge_controls, first_edge_indices, axis=1)
    b021s = tf.gather(cartesian_first_edge_controls, second_edge_indices, axis=1)
    b012s = tf.gather(cartesian_second_edge_controls, second_edge_indices, axis=1)
    b102s = tf.gather(cartesian_first_edge_controls, third_edge_indices, axis=1)
    b201s = tf.gather(cartesian_second_edge_controls, third_edge_indices, axis=1)
    swapped_e1_face_is = tf.cast(tf.squeeze(tf.where(~facetoedgesign_tf[...,0])), tf.int32)
    swapped_e2_face_is = tf.cast(tf.squeeze(tf.where(~facetoedgesign_tf[...,1])), tf.int32)
    swapped_e3_face_is = tf.cast(tf.squeeze(tf.where(~facetoedgesign_tf[...,2])), tf.int32)
    swapped_e1_is = tf.gather(first_edge_indices, swapped_e1_face_is)
    swapped_e2_is = tf.gather(second_edge_indices, swapped_e2_face_is)
    swapped_e3_is = tf.gather(third_edge_indices, swapped_e3_face_is)
    update_indices = tf.stack((tf.broadcast_to(tf.range(tf.shape(b210s)[0])[:,None],(tf.shape(b210s)[0],tf.shape(swapped_e1_face_is)[0])), tf.broadcast_to(swapped_e1_face_is[None,:],(tf.shape(b210s)[0],tf.shape(swapped_e1_face_is)[0]))), axis=-1)
    b210s = tf.tensor_scatter_nd_update(b210s, update_indices, tf.gather(cartesian_second_edge_controls, swapped_e1_is, axis=1))
    b120s = tf.tensor_scatter_nd_update(b120s, update_indices, tf.gather(cartesian_first_edge_controls, swapped_e1_is, axis=1))
    update_indices = tf.stack((tf.broadcast_to(tf.range(tf.shape(b021s)[0])[:,None],(tf.shape(b021s)[0],tf.shape(swapped_e2_face_is)[0])), tf.broadcast_to(swapped_e2_face_is[None,:],(tf.shape(b021s)[0],tf.shape(swapped_e2_face_is)[0]))), axis=-1)
    b021s = tf.tensor_scatter_nd_update(b021s, update_indices, tf.gather(cartesian_second_edge_controls, swapped_e2_is, axis=1))
    b012s = tf.tensor_scatter_nd_update(b012s, update_indices, tf.gather(cartesian_first_edge_controls, swapped_e2_is, axis=1))
    update_indices = tf.stack((tf.broadcast_to(tf.range(tf.shape(b102s)[0])[:,None],(tf.shape(b102s)[0],tf.shape(swapped_e3_face_is)[0])), tf.broadcast_to(swapped_e3_face_is[None,:],(tf.shape(b102s)[0],tf.shape(swapped_e3_face_is)[0]))), axis=-1)
    b102s = tf.tensor_scatter_nd_update(b102s, update_indices, tf.gather(cartesian_second_edge_controls, swapped_e3_is, axis=1))
    b201s = tf.tensor_scatter_nd_update(b201s, update_indices, tf.gather(cartesian_first_edge_controls, swapped_e3_is, axis=1))

    cart_v1_dirs = tf.gather(cartesian_vertex_directions, faces_tf[...,0], axis=1)
    cart_v2_dirs = tf.gather(cartesian_vertex_directions, faces_tf[...,1], axis=1)
    cart_v3_dirs = tf.gather(cartesian_vertex_directions, faces_tf[...,2], axis=1)
    if not tf.math.reduce_any(tf.math.is_nan(b111_barys)):
        b111_uv, b111_u_proportion = tf.split(b111_barys, num_or_size_splits=2, axis=-1)
        b111_u = b111_uv * b111_u_proportion
        b111_v = b111_uv - b111_u
        cartesian_b111_direction = b111_u[...,None]*cart_v1_dirs + b111_v[...,None]*cart_v2_dirs + (1-b111_u-b111_v)[...,None]*cart_v3_dirs
        cartesian_b111_direction /= tf_safe_norm_three(cartesian_b111_direction)[...,None]
    else:
        cartesian_b111_direction = (cart_v1_dirs + cart_v2_dirs + cart_v3_dirs) / 3
        cartesian_b111_direction /= tf_safe_norm_three(cartesian_b111_direction)[...,None]
    b111s = cartesian_b111_direction * other_control_dists[...,2*tf.shape(edges_tf)[0]:,None]
    all_controls = tf.stack((b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s), axis=-2)
    return all_controls

def render_icosahedron(zyx_start, zyx_vertices, zyx_faces, zyx_dists, bbox, normals=None, zyx_controls=None, edges=None, facetoedgemap=None, vertextofacemap=None, facetoedgesign=None, rays=None, return_mesh=False):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    translation = np.array((zmin, ymin, xmin))
    zyx_center = zyx_start - translation
    center = zyx_center[::-1]
    vertices = zyx_center + (zyx_dists+1)[:,None]*np.copy(zyx_vertices).reshape((-1,3))
    vertices = vertices[...,[2,1,0]]
    faces = np.copy(zyx_faces).reshape((-1,3))
    volume = np.zeros((xmax-xmin, ymax-ymin, zmax-zmin), dtype=bool)
    all_control_points = []
    face_vertices = vertices[faces]
    normals_12 = None
    if normals is None:
        normals = get_vertex_normals(vertices, faces, vertextofacemap)
        normals = normals[faces]
    else:
        normals_12 = normals
        normals = normals[faces]
    normals /= np.linalg.norm(normals, axis=2)[...,None]
    if normals_12 is not None:
        normals_12 /= np.linalg.norm(normals_12, axis=1)[...,None]
    if zyx_controls is None:
        all_control_points = get_control_points(face_vertices, normals)
    else:
        all_control_points = zyx_center + zyx_controls
        all_control_points = all_control_points[...,[2,1,0]]
    b300s, b030s, b003s, b012s, b021s, b102s, b201s, b120s, b210s, b111s = [np.squeeze(sub) for sub in np.split(all_control_points, 10, axis=1)]
    all_control_points_reshaped = all_control_points.reshape((-1,3))
    translated_control_points = all_control_points_reshaped-center[None,:]
    outer_bounding_sphere_radius = np.max(np.linalg.norm(translated_control_points, axis=-1))
    planar_face_equations = get_planar_face_equations(vertices, faces)
    interior_planar_face_signs = get_planar_face_signs(planar_face_equations, np.array([center]))
    max_area = get_maximum_planar_face_area(vertices, faces)
    if edges is None:
        _,_, edges, facetoedgemap, vertextofacemap, facetoedgesign = icosahedron()
    subdivisions = 0
    unsubdivided_vertices = np.copy(vertices)
    unsubdivided_faces = np.copy(faces)
    tf_output = zyx_controls is not None
    if tf_output:
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = tf.constant(vertices, tf.float32), tf.constant(faces, tf.int32), tf.cast(edges, tf.int32), tf.cast(facetoedgemap, tf.int32), tf.ragged.constant(vertextofacemap, tf.int32), tf.constant(facetoedgesign)
        barycentric_faces, bary_face_to_unsubbed_face = rays.cached_subdivision_output[0]['barycentric_faces'], rays.cached_subdivision_output[0]['bary_face_to_unsubbed_face']
    while max_area > 0.75 and subdivisions < 4:
        if zyx_controls is not None:
            vertices = subdivide_tri_tf_precomputed(vertices, rays, subdivisions, controls=tf.cast(all_control_points, tf.float32))[0]
            faces = rays.cached_subdivision_output[subdivisions+1]['faces_tf']
            # vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions, barycentric_faces, bary_face_to_unsubbed_face = subdivide_tri_tf(
                # vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, subdivisions, barycentric_faces=barycentric_faces, bary_face_to_unsubbed_face=bary_face_to_unsubbed_face, control_points=tf.cast(all_control_points, tf.float32))
        else:
            vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = subdivide_tri(
                vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign, sub_method="pn", vertex_normals=normals_12)
        normals_12 = None
        if tf_output:
            max_area = get_maximum_planar_face_area(vertices.numpy(), faces.numpy())
        else:
            max_area = get_maximum_planar_face_area(vertices, faces)
        subdivisions +=1
    if tf_output:
        sub_attrs = rays.cached_subdivision_output[subdivisions]
        faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = sub_attrs['faces_tf'], sub_attrs['edges_tf'], sub_attrs['facetoedgemap_tf'], sub_attrs['vertextofacemap_tf'], sub_attrs['facetoedgesign_tf']
        vertices, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = vertices.numpy(), faces.numpy(), edges.numpy(), facetoedgemap.numpy(), vertextofacemap.numpy(), facetoedgesign.numpy()
    volume_indices = np.indices(volume.shape)
    volume_indices = np.moveaxis(volume_indices, 0, -1)
    ps = volume_indices[:, :, :].reshape((-1, 3))
    
    outer_bounding_sphere = ball(np.ceil(outer_bounding_sphere_radius).astype(int), dtype=bool)
    outer_sphere_center = np.array(outer_bounding_sphere.shape) // 2
    z_start = np.maximum(0, outer_sphere_center[0]-center[0])
    z_stop = np.minimum(outer_bounding_sphere.shape[0], volume.shape[0] + z_start)
    y_start = np.maximum(0, outer_sphere_center[1]-center[1])
    y_stop = np.minimum(outer_bounding_sphere.shape[1], volume.shape[1] + y_start)
    x_start = np.maximum(0, outer_sphere_center[2]-center[2])
    x_stop = np.minimum(outer_bounding_sphere.shape[2], volume.shape[2] + x_start)
    outer_sphere_mask = outer_bounding_sphere[z_start:z_stop,y_start:y_stop,x_start:x_stop]
    inside_outer_bounding_sphere = outer_sphere_mask.reshape((-1))
    
    # inner_bounding_sphere_radius = np.min(np.linalg.norm(translated_control_points, axis=-1))
    # inner_bounding_sphere = ball(np.floor(inner_bounding_sphere_radius).astype(int), dtype=bool)
    # inner_sphere_center = np.array(inner_bounding_sphere.shape) // 2
    # mask_z_start = np.maximum(0, -inner_sphere_center[0]+center[0])
    # mask_z_stop = np.minimum(inner_bounding_sphere.shape[0] + mask_z_start, volume.shape[0] + mask_z_start)
    # mask_y_start = np.maximum(0, -inner_sphere_center[1]+center[1])
    # mask_y_stop = np.minimum(inner_bounding_sphere.shape[1] + mask_y_start, volume.shape[1] + mask_y_start)
    # mask_x_start = np.maximum(0, -inner_sphere_center[2]+center[2])
    # mask_x_stop = np.minimum(inner_bounding_sphere.shape[2] + mask_x_start, volume.shape[2] + mask_x_start)
    # z_start = np.maximum(0, inner_sphere_center[0]-center[0])
    # z_stop = np.minimum(inner_bounding_sphere.shape[0], volume.shape[0] + z_start)
    # y_start = np.maximum(0, inner_sphere_center[1]-center[1])
    # y_stop = np.minimum(inner_bounding_sphere.shape[1], volume.shape[1] + y_start)
    # x_start = np.maximum(0, inner_sphere_center[2]-center[2])
    # x_stop = np.minimum(inner_bounding_sphere.shape[2], volume.shape[2] + x_start)
    # inner_sphere_mask = np.zeros(volume.shape, dtype=bool)
    # inner_sphere_mask[mask_z_start:mask_z_stop,mask_y_start:mask_y_stop,mask_x_start:mask_x_stop] = inner_bounding_sphere[z_start:z_stop,y_start:y_stop,x_start:x_stop]
    # inside_inner_bounding_sphere = inner_sphere_mask.reshape((-1))
    # outside_inner_bounding_sphere = ~inside_inner_bounding_sphere

    # volume[tuple(ps[inside_inner_bounding_sphere].T)] = True
    # ps = ps[inside_outer_bounding_sphere & outside_inner_bounding_sphere]
    try:
        ps = ps[inside_outer_bounding_sphere]
    except:
        pass

    # inside_face_planes = inside_tetrahedrons(unsubdivided_vertices, unsubdivided_faces, center, ps)
    # volume[tuple(ps[inside_face_planes].T)] = True
    # ps = ps[~inside_face_planes]
    # do voxelization in batches to avoid running out of vram
    vram_cap_gb = 6
    num_subsets = np.ceil(ps.nbytes * vertices[faces].nbytes * 1e-9 / (3*vram_cap_gb)).astype(int)
    num_subsets = np.maximum(num_subsets, 1)
    face_subsets = np.array_split(faces, num_subsets)
    for face_subset in face_subsets:
        inside_tetras = inside_tetrahedrons(vertices, face_subset, center, ps)
        volume[tuple(ps[inside_tetras].T)] = True
    if return_mesh:
        return volume.transpose(2,1,0), vertices[...,[2,1,0]][edges] - zyx_center[None,None,:]
    else:
        return volume.transpose(2,1,0)