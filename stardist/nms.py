from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import tensorflow as tf
from time import time
from .utils import _normalize_grid
from .bezier_utils import dists_to_volume, bounding_radius_outer, icosahedron_bbox, render_icosahedron, pred_instance_to_control_points, pred_instances_to_control_points, pred_instance_to_control_points, subdivide_tris_tf_bary_one_shot
from .rays3d import reorder_faces

def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b,b),)*prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0]>0 else None,
                         -_bs[1] if _bs[1]>0 else None)  for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


def _non_maximum_suppression_old(coord, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False, max_bbox_search=True):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_rays)

    b: don't use pixel closer than b pixels to the image boundary

    returns retained points
    """
    from .lib.stardist2d import c_non_max_suppression_inds_old

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2
    assert coord.ndim == 4
    grid = _normalize_grid(grid,2)

    # mask = prob > prob_thresh
    # if b is not None and b > 0:
    #     _mask = np.zeros_like(mask)
    #     _mask[b:-b,b:-b] = True
    #     mask &= _mask

    mask = _ind_prob_thresh(prob, prob_thresh, b)

    polygons = coord[mask]
    scores   = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.zeros(len(ind), bool)
    polygons = polygons[ind]
    scores = scores[ind]

    if max_bbox_search:
        # map pixel indices to ids of sorted polygons (-1 => polygon at that pixel not a candidate)
        mapping = -np.ones(mask.shape,np.int32)
        mapping.flat[ np.flatnonzero(mask)[ind] ] = range(len(ind))
    else:
        mapping = np.empty((0,0),np.int32)

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds_old(np.ascontiguousarray(polygons.astype(np.int32)),
                    mapping, np.float32(nms_thresh), np.int32(max_bbox_search),
                    np.int32(grid[0]), np.int32(grid[1]),np.int32(verbose))

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))
        print("NMS took %.4f s" % (time() - t))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)],axis=-1)
    return points


def non_maximum_suppression(dist, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5,
                            use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 2D polygons

    Retains only polygons whose overlap is smaller than nms_thresh

    dist.shape = (Ny,Nx, n_rays)
    prob.shape = (Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression(dist, prob, ....

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2 and dist.ndim == 3  and prob.shape == dist.shape[:2]
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    n_rays = dist.shape[-1]

    grid = _normalize_grid(grid,2)

    # mask = prob > prob_thresh
    # if b is not None and b > 0:
    #     _mask = np.zeros_like(mask)
    #     _mask[b:-b,b:-b] = True
    #     mask &= _mask

    mask = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(mask), axis=1)

    dist   = dist[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    dist   = dist[ind]
    scores = scores[ind]
    points = points[ind]

    points = (points * np.array(grid).reshape((1,2)))

    if verbose:
        t = time()

    inds = non_maximum_suppression_inds(dist, points.astype(np.int32, copy=False), scores=scores,
                                        use_bbox=use_bbox, use_kdtree=use_kdtree,
                                        thresh=nms_thresh, verbose=verbose)

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return points[inds], scores[inds], dist[inds]


def non_maximum_suppression_sparse(dist, prob, points, b=2, nms_thresh=0.5,
                                   use_bbox=True, use_kdtree = True, verbose=False):
    """Non-Maximum-Supression of 2D polygons from a list of dists, probs (scores), and points

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,2)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)
    n_rays = dist.shape[-1]

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        points.shape[-1]==2 and len(prob) == len(dist) == len(points)

    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    if verbose:
        print("non-maximum suppression...")
        t = time()

    inds = non_maximum_suppression_inds(disti, pointsi, scores=probi, thresh=nms_thresh, use_kdtree = use_kdtree, verbose=verbose)

    if verbose:
        print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return pointsi[inds], probi[inds], disti[inds], inds_original[inds]


def non_maximum_suppression_inds(dist, points, scores, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1):
    """
    Applies non maximum supression to ray-convex polygons given by dists and points
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 2)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """

    from .lib.stardist2d import c_non_max_suppression_inds

    assert dist.ndim == 2
    assert points.ndim == 2

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    inds = c_non_max_suppression_inds(_prep(dist,  np.float32),
                                      _prep(points, np.float32),
                                      int(use_kdtree),
                                      int(use_bbox),
                                      int(verbose),
                                      np.float32(thresh))

    return inds


#########


def non_maximum_suppression_3d(dist, prob, rays, grid=(1,1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 3D polyhedra

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (Nz,Ny,Nx, n_rays)
    prob.shape = (Nz,Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression_3d(dist, prob, ....
    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)

    assert prob.ndim == 3 and dist.ndim == 4 and dist.shape[-1] == len(rays) and prob.shape == dist.shape[:3]

    grid = _normalize_grid(grid,3)

    verbose and print("predicting instances with prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}".format(prob_thresh=prob_thresh, nms_thresh=nms_thresh), flush=True)

    # ind_thresh = prob > prob_thresh
    # if b is not None and b > 0:
    #     _ind_thresh = np.zeros_like(ind_thresh)
    #     _ind_thresh[b:-b,b:-b,b:-b] = True
    #     ind_thresh &= _ind_thresh

    ind_thresh = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(ind_thresh), axis=1)
    verbose and print("found %s candidates"%len(points))
    probi = prob[ind_thresh]
    disti = dist[ind_thresh]

    _sorted = np.argsort(probi)[::-1]
    probi = probi[_sorted]
    disti = disti[_sorted]
    points = points[_sorted]

    verbose and print("non-maximum suppression...")
    points = (points * np.array(grid).reshape((1,3)))

    inds = non_maximum_suppression_3d_inds(disti, points, rays=rays, scores=probi, thresh=nms_thresh,
                                           use_bbox=use_bbox, use_kdtree = use_kdtree,
                                           verbose=verbose)

    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
    return points[inds], probi[inds], disti[inds]


def non_maximum_suppression_3d_sparse(dist, prob, points, rays, b=2, nms_thresh=0.5, use_kdtree = True, verbose=False):
    """Non-Maximum-Supression of 3D polyhedra from a list of dists, probs and points

    Retains only polyhedra whose overlap is smaller than nms_thresh
    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,3)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...
    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        dist.shape[-1] == len(rays) and points.shape[-1]==3 and len(prob) == len(dist) == len(points)

    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    verbose and print("non-maximum suppression...")

    inds = non_maximum_suppression_3d_inds(disti, pointsi, rays=rays, scores=probi, thresh=nms_thresh, use_kdtree = use_kdtree, verbose=verbose)

    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
    return pointsi[inds], probi[inds], disti[inds], inds_original[inds]


def non_maximum_suppression_3d_inds(dist, points, rays, scores, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1):
    """
    Applies non maximum supression to ray-convex polyhedra given by dists and rays
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 3)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """
    from .lib.stardist3d import c_non_max_suppression_inds

    assert dist.ndim == 2
    assert points.ndim == 2
    assert dist.shape[1] == len(rays)

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.ones(n_poly, bool)
    dist = dist[ind]
    points = points[ind]
    scores = scores[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds(_prep(dist, np.float32),
                                                _prep(points, np.float32),
                                                _prep(rays.vertices, np.float32),
                                                _prep(rays.faces, np.int32),
                                                _prep(scores, np.float32),
                                                int(use_bbox),
                                                int(use_kdtree),
                                                int(verbose),
                                                np.float32(thresh))

    if verbose:
        print("NMS took %.4f s" % (time() - t))

    return survivors

def non_maximum_suppression_patch(dist, prob, rays, grid=(1,1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, use_bbox=True, use_kdtree=True, verbose=False, return_meshes=False):
    """Non-Maximum-Supression of 3D polyhedra

    Retains only polyhedra whose overlap is smaller than nms_thresh

    dist.shape = (Nz,Ny,Nx, n_rays)
    prob.shape = (Nz,Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression_3d(dist, prob, ....
    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)

    assert prob.ndim == 3 and dist.ndim == 4 and (dist.shape[-1] == 3*len(rays)+3*len(rays.faces)+2*len(rays.edges) \
    or dist.shape[-1] == len(rays)+len(rays.faces)+2*len(rays.edges)) and prob.shape == dist.shape[:3]

    grid = _normalize_grid(grid,3)

    verbose and print("predicting instances with prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}".format(prob_thresh=prob_thresh, nms_thresh=nms_thresh), flush=True)

    # ind_thresh = prob > prob_thresh
    # if b is not None and b > 0:
    #     _ind_thresh = np.zeros_like(ind_thresh)
    #     _ind_thresh[b:-b,b:-b,b:-b] = True
    #     ind_thresh &= _ind_thresh

    ind_thresh = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(ind_thresh), axis=1)
    verbose and print("found %s candidates"%len(points))
    probi = prob[ind_thresh]
    disti = dist[ind_thresh]

    _sorted = np.argsort(probi)[::-1]
    probi = probi[_sorted]
    disti = disti[_sorted]
    points = points[_sorted]

    verbose and print("non-maximum suppression...")
    points = (points * np.array(grid).reshape((1,3)))

    nms_result = non_maximum_suppression_patch_inds(disti, points, rays=rays, scores=probi, img_shape=dist.shape[:3], thresh=nms_thresh,
                                           use_bbox=use_bbox, use_kdtree = use_kdtree,
                                           verbose=verbose, return_meshes=return_meshes)
    if return_meshes:
        inds, meshes = nms_result
    else:
        inds = nms_result
    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))

    if return_meshes:
        return points[inds], probi[inds], disti[inds], meshes
    else:
        return points[inds], probi[inds], disti[inds]


def non_maximum_suppression_patch_sparse(dist, prob, points, rays, img_shape, b=2, nms_thresh=0.5, use_kdtree = True, verbose=False, return_meshes=False):
    """Non-Maximum-Supression of 3D polyhedra from a list of dists, probs and points

    Retains only polyhedra whose overlap is smaller than nms_thresh
    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,3)

    returns the retained instances

    (pointsi, probi, disti, indsi)

    with
    pointsi = points[indsi] ...
    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)

    print(dist.shape)
    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        (dist.shape[-1] == 3*len(rays)+3*len(rays.faces)+2*len(rays.edges) \
        or dist.shape[-1] == len(rays)+len(rays.faces)+2*len(rays.edges)) \
        and points.shape[-1]==3 and len(prob) == len(dist) == len(points)

    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    inds_original = np.arange(len(prob))
    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]
    inds_original = inds_original[_sorted]

    verbose and print("non-maximum suppression...")
    nms_result = non_maximum_suppression_patch_inds(disti, pointsi, rays=rays, scores=probi, img_shape=img_shape, thresh=nms_thresh, use_kdtree = use_kdtree, verbose=verbose, return_meshes=return_meshes)
    if return_meshes:
        inds, meshes = nms_result
    else:
        inds = nms_result
    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))

    if return_meshes:
        return pointsi[inds], probi[inds], disti[inds], inds_original[inds], meshes
    else:
        return pointsi[inds], probi[inds], disti[inds], inds_original[inds]

def non_maximum_suppression_patch_inds(dist, points, rays, scores, img_shape, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1, return_meshes=False):
    """
    Applies non maximum supression to ray-convex polyhedra given by dists and rays
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 3)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """
    from .lib.patchdist import c_non_max_suppression_inds

    assert dist.ndim == 2
    assert points.ndim == 2
    assert dist.shape[1] == 3*len(rays)+3*len(rays.faces)+2*len(rays.edges) or dist.shape[1] == len(rays)+len(rays.faces)+2*len(rays.edges)

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.ones(n_poly, bool)
    dist = dist[ind]
    points = points[ind]
    scores = scores[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    if verbose:
        t = time()

    print("initial 3d nms candidates:", survivors.size)
    start_time_3d = time()
    if dist.shape[0] != 0:
        if dist.shape[1] == 3*len(rays)+3*len(rays.faces)+2*len(rays.edges):
            split_indices = len(rays), 2*len(rays), 3*len(rays), 3*len(rays)+2*len(rays.faces)
            dists, thetas, phis, b111_barys, other_control_dists = np.split(dist, indices_or_sections=split_indices, axis=-1)
            cartesian_vertices = rays.voronai_vertices_to_unit_vertices_tf(thetas, phis).numpy()
            max_dist = 0
            max_theta = 0
            max_phi = 0
            for dist, theta, phi in zip(dists, thetas, phis):
                if np.max(dist)>max_dist:
                    max_dist=np.max(dist)
                if np.max(theta)>max_theta:
                    max_theta=np.max(theta)
                if np.max(phi)>max_phi:
                    max_phi=np.max(phi)
            print("MAX dist",max_dist,"MAX theta",max_theta,"MAX phi",max_phi)
        else:
            split_indices = len(rays),
            dists, other_control_dists = np.split(dist, indices_or_sections=split_indices, axis=-1)
            cartesian_vertices = np.broadcast_to(rays.vertices, (np.prod(dists.shape[:-1]), len(rays), 3))
            cartesian_vertices = cartesian_vertices.reshape(tuple(dists.shape) + (3,))
            b111_barys = tf.constant(((np.nan,np.nan),), tf.float32)

        control_points = pred_instances_to_control_points(tf.constant(cartesian_vertices), tf.constant(dists), tf.constant(other_control_dists), b111_barys, rays.edges_tf, rays.faces_tf, rays.facetoedgemap_tf, rays.facetoedgesign_tf)
        addl_subdivided_vertices = subdivide_tris_tf_bary_one_shot(control_points, rays.cached_subdivision_output[4]['all_addl_bary_unsubbed_faces'], rays.cached_subdivision_output[4]['all_addl_bary_vertices']).numpy()
        addl_subdivided_vertex_dists = np.linalg.norm(addl_subdivided_vertices, axis=-1)
        addl_subdivided_vertex_dirs = addl_subdivided_vertices / addl_subdivided_vertex_dists[...,None]
        cartesian_vertices = np.concatenate((cartesian_vertices, addl_subdivided_vertex_dirs), axis=-2)
        dists = np.concatenate((dists, addl_subdivided_vertex_dists), axis=-1)
        subdivided_faces = rays.cached_subdivision_output[4]['faces_tf'].numpy().astype(np.int64)
        subdivided_faces = np.array(reorder_faces(cartesian_vertices[0],subdivided_faces))

        survivors[ind] = c_non_max_suppression_inds(_prep(dists, np.float32),
                                                    _prep(points, np.float32),
                                                    _prep(cartesian_vertices, np.float32),
                                                    _prep(subdivided_faces, np.int32),
                                                    _prep(scores, np.float32),
                                                    int(use_bbox),
                                                    int(use_kdtree),
                                                    int(verbose),
                                                    np.float32(thresh))
    end_time_3d = time()
    print("time for 3d nms:",end_time_3d - start_time_3d,"s")
    if verbose:
        print("NMS took %.4f s" % (time() - t))
    if return_meshes:
        subdivided_facetoedgemap = rays.cached_subdivision_output[4]['facetoedgemap_tf'].numpy()
        subdivided_edges = rays.cached_subdivision_output[4]['edges_tf'].numpy()
        icosa_meshes = np.take(cartesian_vertices * dists[...,None], subdivided_edges, axis=-2)
        icosa_meshes += points[:,None,None,:]
        return survivors, icosa_meshes[survivors]
    else:
        return survivors
