"""
Ray factory

classes that provide vertex and triangle information for rays on spheres

Example:

    rays = Rays_Tetra(n_level = 4)

    print(rays.vertices)
    print(rays.faces)

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import math
import numpy as np
import tensorflow as tf
from scipy.spatial import ConvexHull, SphericalVoronoi, geometric_slerp
import copy
import warnings
from .bezier_utils import icosahedron, subdivide_tri, subdivide_tri_tf, subdivide_tri_tf_bary, tf_safe_norm, dists_to_controls_tf

class Rays_Base(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._vertices, self._faces = self.setup_vertices_faces()
        self._vertices = np.asarray(self._vertices, np.float32)
        self._faces = np.asarray(self._faces, int)
        self._faces = np.asanyarray(self._faces)

    def setup_vertices_faces(self):
        """has to return

         verts , faces

         verts = ( (z_1,y_1,x_1), ... )
         faces ( (0,1,2), (2,3,4), ... )

         """
        raise NotImplementedError()

    @property
    def vertices(self):
        """read-only property"""
        return self._vertices.copy()

    @property
    def faces(self):
        """read-only property"""
        return self._faces.copy()

    def __getitem__(self, i):
        return self.vertices[i]

    def __len__(self):
        return len(self._vertices)

    def __repr__(self):
        def _conv(x):
            if isinstance(x,(tuple, list, np.ndarray)):
                return "_".join(_conv(_x) for _x in x)
            if isinstance(x,float):
                return "%.2f"%x
            return str(x)
        return "%s_%s" % (self.__class__.__name__, "_".join("%s_%s" % (k, _conv(v)) for k, v in sorted(self.kwargs.items())))
    
    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": self.kwargs
        }

    def dist_loss_weights(self, anisotropy = (1,1,1)):
        """returns the anisotropy corrected weights for each ray"""
        anisotropy = np.array(anisotropy)
        assert anisotropy.shape == (3,)
        return np.linalg.norm(self.vertices*anisotropy, axis = -1)

    def volume(self, dist=None):
        """volume of the starconvex polyhedron spanned by dist (if None, uses dist=1)
        dist can be a nD array, but the last dimension has to be of length n_rays
        """
        if dist is None: dist = np.ones_like(self.vertices)

        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
        d = np.linalg.det(list(vs)).reshape((len(self.faces),)+dist.shape[1:-1])
        
        return -1./6*np.sum(d, axis = 0)
    
    def surface(self, dist=None):
        """surface area of the starconvex polyhedron spanned by dist (if None, uses dist=1)"""
        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")

        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
       
        pa = vs[...,1,:]-vs[...,0,:]
        pb = vs[...,2,:]-vs[...,0,:]

        d = .5*np.linalg.norm(np.cross(list(pa), list(pb)), axis = -1)
        d = d.reshape((len(self.faces),)+dist.shape[1:-1])
        return np.sum(d, axis = 0)

    
    def copy(self, scale=(1,1,1)):
        """ returns a copy whose vertices are scaled by given factor"""
        scale = np.asarray(scale)
        assert scale.shape == (3,)
        res = copy.deepcopy(self)
        res._vertices *= scale[np.newaxis]
        return res 



    
def rays_from_json(d):
    return eval(d["name"])(**d["kwargs"])


################################################################

class Rays_Explicit(Rays_Base):
    def __init__(self, vertices0, faces0):
        self.vertices0, self.faces0 = vertices0, faces0
        super().__init__(vertices0=list(vertices0), faces0=list(faces0))
        
    def setup_vertices_faces(self):
        return self.vertices0, self.faces0
    

class Rays_Cartesian(Rays_Base):
    def __init__(self, n_rays_x=11, n_rays_z=5):
        super().__init__(n_rays_x=n_rays_x, n_rays_z=n_rays_z)

    def setup_vertices_faces(self):
        """has to return list of ( (z_1,y_1,x_1), ... )  _"""
        n_rays_x, n_rays_z = self.kwargs["n_rays_x"], self.kwargs["n_rays_z"]
        dphi = np.float32(2. * np.pi / n_rays_x)
        dtheta = np.float32(np.pi / n_rays_z)

        verts = []
        for mz in range(n_rays_z):
            for mx in range(n_rays_x):
                phi = mx * dphi
                theta = mz * dtheta
                if mz == 0:
                    theta = 1e-12
                if mz == n_rays_z - 1:
                    theta = np.pi - 1e-12
                dx = np.cos(phi) * np.sin(theta)
                dy = np.sin(phi) * np.sin(theta)
                dz = np.cos(theta)
                if mz == 0 or mz == n_rays_z - 1:
                    dx += 1e-12
                    dy += 1e-12
                verts.append([dz, dy, dx])

        verts = np.array(verts)

        def _ind(mz, mx):
            return mz * n_rays_x + mx

        faces = []

        for mz in range(n_rays_z - 1):
            for mx in range(n_rays_x):
                faces.append([_ind(mz, mx), _ind(mz + 1, (mx + 1) % n_rays_x), _ind(mz, (mx + 1) % n_rays_x)])
                faces.append([_ind(mz, mx), _ind(mz + 1, mx), _ind(mz + 1, (mx + 1) % n_rays_x)])

        faces = np.array(faces)

        return verts, faces


class Rays_SubDivide(Rays_Base):
    """
    Subdivision polyehdra

    n_level = 1 -> base polyhedra
    n_level = 2 -> 1x subdivision
    n_level = 3 -> 2x subdivision
                ...
    """

    def __init__(self, n_level=4):
        super().__init__(n_level=n_level)

    def base_polyhedron(self):
        raise NotImplementedError()

    def setup_vertices_faces(self):
        n_level = self.kwargs["n_level"]
        verts0, faces0 = self.base_polyhedron()
        return self._recursive_split(verts0, faces0, n_level)

    def _recursive_split(self, verts, faces, n_level):
        if n_level <= 1:
            return verts, faces
        else:
            verts, faces = Rays_SubDivide.split(verts, faces)
            return self._recursive_split(verts, faces, n_level - 1)

    @classmethod
    def split(self, verts0, faces0):
        """split a level"""

        split_edges = dict()
        verts = list(verts0[:])
        faces = []

        def _add(a, b):
            """ returns index of middle point and adds vertex if not already added"""
            edge = tuple(sorted((a, b)))
            if not edge in split_edges:
                v = .5 * (verts[a] + verts[b])
                v *= 1. / np.linalg.norm(v)
                verts.append(v)
                split_edges[edge] = len(verts) - 1
            return split_edges[edge]

        for v1, v2, v3 in faces0:
            ind1 = _add(v1, v2)
            ind2 = _add(v2, v3)
            ind3 = _add(v3, v1)
            faces.append([v1, ind1, ind3])
            faces.append([v2, ind2, ind1])
            faces.append([v3, ind3, ind2])
            faces.append([ind1, ind2, ind3])

        return verts, faces


class Rays_Tetra(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal tetrahedron (4 vertices)
    n_level = 2 -> 1x subdivision (10 vertices)
    n_level = 3 -> 2x subdivision (34 vertices)
                ...
    """

    def base_polyhedron(self):
        verts = np.array([
            [np.sqrt(8. / 9), 0., -1. / 3],
            [-np.sqrt(2. / 9), np.sqrt(2. / 3), -1. / 3],
            [-np.sqrt(2. / 9), -np.sqrt(2. / 3), -1. / 3],
            [0., 0., 1.]
        ])
        faces = [[0, 1, 2],
                 [0, 3, 1],
                 [0, 2, 3],
                 [1, 3, 2]]

        return verts, faces


class Rays_Octo(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal Octahedron (6 vertices)
    n_level = 2 -> 1x subdivision (18 vertices)
    n_level = 3 -> 2x subdivision (66 vertices)

    """

    def base_polyhedron(self):
        verts = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0]])

        faces = [[0, 1, 4],
                 [0, 5, 1],
                 [1, 2, 4],
                 [1, 5, 2],
                 [2, 3, 4],
                 [2, 5, 3],
                 [3, 0, 4],
                 [3, 5, 0],
                 ]

        return verts, faces


def reorder_faces(verts, faces):
    """reorder faces such that their orientation points outward"""
    def _single(face):
        return face[::-1] if np.linalg.det(verts[face])>0 else face
    return tuple(map(_single, faces))


class Rays_GoldenSpiral(Rays_Base):
    def __init__(self, n=70, anisotropy = None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy))

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)

        # the smaller golden angle = 2pi * 0.3819...
        g = (3. - np.sqrt(5.)) * np.pi
        phi = g * np.arange(n)
        # z = np.linspace(-1, 1, n + 2)[1:-1]
        # rho = np.sqrt(1. - z ** 2)
        # verts = np.stack([rho*np.cos(phi), rho*np.sin(phi),z]).T
        #
        z = np.linspace(-1, 1, n)
        rho = np.sqrt(1. - z ** 2)
        verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T

        # warnings.warn("ray definition has changed! Old results are invalid!")

        # correct for anisotropy
        verts = verts/anisotropy
        #verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        hull = ConvexHull(verts)
        faces = reorder_faces(verts,hull.simplices)

        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        return verts, faces

class Rays_Patch(Rays_Base):
    def __init__(self, n=70, anisotropy = None, subdivisions=None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy), subdivisions=subdivisions)

    def setup_other(self, faces, verts):
        edges = []
        facetoedgemap = [[] for face in faces]
        vertextofacemap = [[] for vert in verts]
        for face_index, face in enumerate(faces):
            for vertex in face:
                edge = sorted([other_vertex for other_vertex in face if other_vertex != vertex])
                edge_index = None
                for seen_edge_index, seen_edge in enumerate(edges):
                    if seen_edge == edge:
                        edge_index = seen_edge_index
                        break
                if edge_index is None:
                    edges.append(edge)
                    edge_index = len(edges) - 1
                facetoedgemap[face_index].append(edge_index)
                vertextofacemap[vertex].append(face_index)
            facetoedgemap[face_index] = [facetoedgemap[face_index][2], facetoedgemap[face_index][0], facetoedgemap[face_index][1]]
        facetoedgesign = []
        for face_index, face in enumerate(faces):
            facetoedgesign.append([edges[facetoedgemap[face_index][0]][0] == face[0], edges[facetoedgemap[face_index][1]][0] == face[1], edges[facetoedgemap[face_index][2]][0] == face[2]])
        self.edges = np.array(edges)
        self.facetoedgemap = np.array(facetoedgemap)
        self.vertextofacemap = vertextofacemap
        self.facetoedgesign = facetoedgesign

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)
        self.anisotropy = anisotropy

        if n == 12:
            verts, faces, self.edges, self.facetoedgemap, self.vertextofacemap, self.facetoedgesign = icosahedron()

            if self.kwargs.get('subdivisions') is not None:
                for _ in range(self.kwargs.get('subdivisions')):
                    verts, faces, edges, facetoedgemap, vertextofacemap, facetoedgesign = subdivide_tri(
                        verts, faces, self.edges, self.facetoedgemap, self.vertextofacemap, self.facetoedgesign, sub_method="pn", vertex_normals=None)
                    self.edges, self.facetoedgemap, self.vertextofacemap, self.facetoedgesign = edges, facetoedgemap, vertextofacemap, facetoedgesign
                    verts = np.copy(verts)[:,[2,1,0]]
            # correct for anisotropy
            verts /= anisotropy
            verts /= np.linalg.norm(verts, axis=-1, keepdims=True)
        else:
            # use golden spirals
            g = (3. - np.sqrt(5.)) * np.pi
            phi = g * np.arange(n)
            z = np.linspace(-1, 1, n)
            rho = np.sqrt(1. - z ** 2)
            verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T
            if n == 6:
                verts = np.array([
                    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
                ], dtype=float)
            verts /= anisotropy
            hull = ConvexHull(verts)
            faces = np.array(reorder_faces(verts,hull.simplices))
            verts /= np.linalg.norm(verts, axis=-1, keepdims=True)
            centroid = np.mean(verts, axis=0)
            for face_index, face in enumerate(faces):
                a, b, c = verts[face]
                normal = np.cross(a-b, c-b)
                normal /= np.linalg.norm(normal)
                face_center = np.mean(verts[face], axis=0)
                k = -np.sum(normal*face_center)
                centroid_sign = np.sign(np.sum(normal*centroid) + k)
                normal_sign = np.sign(np.sum(normal*(face_center+normal)) + k)
                if centroid_sign == normal_sign:
                    faces[face_index] = face[::-1]
            self.setup_other(faces, verts)


        verts_voronai = SphericalVoronoi(verts)
        verts_voronai.sort_vertices_of_regions()
        self.verts_voronai = verts_voronai
        self.verts_voronai.boundaries = []
        self.verts_voronai.boundary_points= []
        self.verts_voronai.boundary_dihedral_angles = []

        def get_plane_normal(edge_vertex, ray_vertex):
            center = np.array((0.,0.,0.))
            v1 = edge_vertex - ray_vertex
            v2 = center - ray_vertex
            return np.cross(v1, v2)
        def get_dihedral_angle(edge_vertex_1, edge_vertex_2, ray_vertex):
            plane_normal_1 = get_plane_normal(edge_vertex_1, ray_vertex)
            plane_normal_2 = get_plane_normal(edge_vertex_2, ray_vertex)
            cos_dihedral = np.dot(plane_normal_1, plane_normal_2) / (np.linalg.norm(plane_normal_1) * np.linalg.norm(plane_normal_2))
            return np.arccos(cos_dihedral)

        t_vals = np.linspace(0, 1, 2000)
        for region_i in range(len(verts_voronai.regions)):
            region = verts_voronai.regions[region_i]
            boundary = []
            boundary_points = []
            boundary_dihedral_angles = []
            n = len(region)
            last_start = None
            last_angle = 0.
            for i in range(n + 1):
                start = verts_voronai.vertices[region][i % n]
                end = verts_voronai.vertices[region][(i + 1) % n]
                boundary_line = geometric_slerp(start, end, t_vals)
                spherical_boundary_line = self.cartesian_to_spherical(boundary_line)
                boundary.extend(spherical_boundary_line)
                boundary_points.append(start)
                dihedral_angle = 0.
                if i != 0:
                    dihedral_angle = get_dihedral_angle(start, last_start, verts[region_i])
                boundary_dihedral_angles.append(dihedral_angle + last_angle)
                last_start = start
                last_angle += dihedral_angle
            boundary_dihedral_angles[-1] = 2 * np.pi
            boundary = np.array(boundary)
            boundary_points = np.array(boundary_points)
            boundary_dihedral_angles = np.array(boundary_dihedral_angles)
            self.verts_voronai.boundaries.append(boundary)
            self.verts_voronai.boundary_points.append(boundary_points)
            self.verts_voronai.boundary_dihedral_angles.append(boundary_dihedral_angles)

        self.vertices_tf = tf.constant(verts, tf.float32)
        self.faces_tf = tf.constant(faces, tf.int32)
        self.edges_tf = tf.constant(self.edges, tf.int32)
        self.facetoedgemap_tf = tf.constant(self.facetoedgemap, tf.int32)
        self.vertextofacemap_tf = tf.ragged.constant(self.vertextofacemap, tf.int32)
        self.facetoedgesign_tf = tf.constant(self.facetoedgesign)
        # not needed anymore with slerp voronai_vertices_to_unit_vertices_tf approach
        self.verts_voronai.boundaries_tf = tf.ragged.constant(self.verts_voronai.boundaries, tf.float32)
        # pad with infinity to avoid ragged tensor annoyance; okay while only used for voronai_vertices_to_unit_vertices_tf
        self.verts_voronai.boundary_points_tf = tf.ragged.constant(self.verts_voronai.boundary_points, tf.float32).to_tensor(default_value=np.inf)
        self.verts_voronai.boundary_dihedral_angles_tf = tf.ragged.constant(self.verts_voronai.boundary_dihedral_angles, tf.float32).to_tensor(default_value=np.inf)

        def subd_output_dict(subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign, subdivided_barycentric_faces, subdivided_barycentric_edges, subdivided_bary_face_to_unsubbed_face):
            return {
                'faces_tf': subdivided_faces,
                'edges_tf': subdivided_edges,
                'facetoedgemap_tf': subdivided_facetoedgemap,
                'vertextofacemap_tf': subdivided_vertextofacemap,
                'facetoedgesign_tf': subdivided_facetoedgesign,
                'barycentric_faces': subdivided_barycentric_faces,
                'barycentric_edges': subdivided_barycentric_edges,
                'bary_face_to_unsubbed_face': subdivided_bary_face_to_unsubbed_face
            }
        subdivided_vertices, subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign = self.vertices_tf, self.faces_tf, self.edges_tf, self.facetoedgemap_tf, self.vertextofacemap_tf, self.facetoedgesign_tf
        subdivided_barycentric_faces = [((1.,0.),(0.,1.),(0.,0.)) for face in faces]
        subdivided_barycentric_edges = [(((1.,0.),(0.,1.)),((0.,1.),(0.,0.)), ((0.,0.),(1.,0.))) for face in faces]
        subdivided_bary_face_to_unsubbed_face = [face_index for face_index in range(len(faces))]
        subdivided_barycentric_faces, subdivided_barycentric_edges, subdivided_bary_face_to_unsubbed_face = tf.constant(subdivided_barycentric_faces, tf.float32), tf.constant(subdivided_barycentric_edges), tf.constant(subdivided_bary_face_to_unsubbed_face)
        self.cached_subdivision_output = {0: subd_output_dict(subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign, subdivided_barycentric_faces, subdivided_barycentric_edges, subdivided_bary_face_to_unsubbed_face)}
        subdivisions = 0
        while subdivisions < 4:
            subdivided_vertices, subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign, subdivisions, subdivided_barycentric_faces, subdivided_barycentric_edges, subdivided_bary_face_to_unsubbed_face \
                = subdivide_tri_tf(subdivided_vertices, subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign, subdivisions, barycentric_faces=subdivided_barycentric_faces, barycentric_edges=subdivided_barycentric_edges, bary_face_to_unsubbed_face=subdivided_bary_face_to_unsubbed_face)
            self.cached_subdivision_output[subdivisions] = subd_output_dict(subdivided_faces, subdivided_edges, subdivided_facetoedgemap, subdivided_vertextofacemap, subdivided_facetoedgesign, subdivided_barycentric_faces, subdivided_barycentric_edges, subdivided_bary_face_to_unsubbed_face)
        control_points = dists_to_controls_tf(self.vertices_tf, self.faces_tf, tf.ones((len(verts))), self.vertextofacemap_tf)
        all_addl_bary_vertices = []
        all_addl_bary_unsubbed_faces = []
        for s in range(4):
            subdivided_facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face = self.cached_subdivision_output[s]['facetoedgemap_tf'], self.cached_subdivision_output[s]['barycentric_faces'], self.cached_subdivision_output[s]['barycentric_edges'], self.cached_subdivision_output[s]['bary_face_to_unsubbed_face']
            _, sub_bary_vertices, sub_bary_unsubbed_faces = subdivide_tri_tf_bary(control_points, subdivided_facetoedgemap, barycentric_faces, barycentric_edges, bary_face_to_unsubbed_face, return_barys=True)
            all_addl_bary_vertices.extend(sub_bary_vertices.numpy().tolist())
            all_addl_bary_unsubbed_faces.extend(sub_bary_unsubbed_faces.numpy().tolist())
            self.cached_subdivision_output[s+1]['all_addl_bary_vertices'] = tf.constant(all_addl_bary_vertices)
            self.cached_subdivision_output[s+1]['all_addl_bary_unsubbed_faces'] = tf.constant(all_addl_bary_unsubbed_faces)
        return verts, faces

    def cartesian_to_spherical(self, cartesian):
        spherical = []
        for point in cartesian:
            z,y,x = point
            r = np.linalg.norm(point)
            theta = np.arccos(z/r)
            phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
            if y==0 and x==0:
                phi = 0.
            theta %= 2*np.pi
            phi %= 2*np.pi
            if theta < 0:
                theta += 2*np.pi
            if phi < 0:
                phi += 2*np.pi
            spherical.append((theta,phi))
        return np.array(spherical)
    def spherical_to_cartesian(self, spherical):
        cartesian = []
        for point in spherical:
            theta, phi = point
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            cartesian.append((z,y,x))
        return np.array(cartesian)
    # 0 <= theta <= 2*pi, 0 <= phi <= 1
    def vertex_voronai_to_unit_vertex(self, vertex_index, theta, phi):
        vertex = self.vertices[vertex_index]
        spherical_vertex = self.cartesian_to_spherical([vertex])[0]
        boundary = self.verts_voronai.boundaries[vertex_index]
        relative_boundary_theta = np.linspace(0, 2*np.pi, len(boundary))
        spherical_edge_point = boundary[np.argmin(np.abs(relative_boundary_theta - theta))]
        edge_point = self.spherical_to_cartesian([spherical_edge_point])[0]
        new_theta = spherical_vertex[0] + phi*(edge_point[0] - spherical_vertex[0])
        new_phi = spherical_vertex[1] + phi*(edge_point[1] - spherical_vertex[1])
        spherical_new_vertex = np.array((new_theta, new_phi))
        unit_vertex = self.spherical_to_cartesian([spherical_new_vertex])[0]
        return unit_vertex
    
    def cartesian_to_spherical_tf(self, cartesian):
        z_s,y_s,x_s = tf.unstack(cartesian, axis=-1)
        r_s = tf_safe_norm(cartesian)
        theta_s = tf.math.acos(z_s/r_s)
        cos_phi_s = tf.math.divide_no_nan(x_s, tf.math.sqrt(x_s**2+y_s**2))
        cos_phi_s = tf.clip_by_value(cos_phi_s, -1.+tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        phi_s = tf.math.sign(y_s)*tf.math.acos(cos_phi_s)
        phi_s = tf.math.multiply_no_nan(phi_s, tf.cast(~tf.logical_and(y_s==0,x_s==0), tf.float32))
        theta_s %= 2*np.pi
        phi_s %= 2*np.pi
        theta_s = theta_s + 2*np.pi*tf.cast(tf.less(theta_s, 0), tf.float32)
        phi_s = phi_s + 2*np.pi*tf.cast(tf.less(phi_s, 0), tf.float32)
        # handle zero inputs
        theta_update_indices = tf.where(tf.math.is_nan(theta_s))
        theta_updates = tf.zeros_like(tf.gather_nd(theta_s, theta_update_indices))
        phi_update_indices = tf.where(tf.math.is_nan(phi_s))
        phi_updates = tf.zeros_like(tf.gather_nd(phi_s, phi_update_indices))
        theta_s = tf.tensor_scatter_nd_update(theta_s, theta_update_indices, theta_updates)
        phi_s = tf.tensor_scatter_nd_update(phi_s, phi_update_indices, phi_updates)
        tf.debugging.assert_all_finite(theta_s,"theta_s ctstf")
        tf.debugging.assert_all_finite(phi_s,"phi_s xtstf")
        return tf.stack([theta_s, phi_s], axis=-1)
    def spherical_to_cartesian_tf(self, spherical):
        tf.debugging.assert_all_finite(spherical,"stctf in")
        theta_s, phi_s = tf.unstack(spherical, axis=-1)
        x_s = tf.math.sin(theta_s)*tf.math.cos(phi_s)
        y_s = tf.math.sin(theta_s)*tf.math.sin(phi_s)
        z_s = tf.math.cos(theta_s)
        tf.debugging.assert_all_finite(tf.stack([z_s,y_s,x_s], axis=-1),"stctf out")
        return tf.stack([z_s,y_s,x_s], axis=-1)
    # 0 <= theta <= 2*pi, 0 <= phi <= 1
    def tf_geometric_slerp(self, p0_s, p1_s, t_s, omegas):
        omega_sines = tf.math.sin(omegas)
        first_term = tf.math.sin((1.-t_s)*omegas) / omega_sines
        first_term = first_term[...,None] * p0_s
        second_term = tf.math.sin(t_s * omegas) / omega_sines
        second_term = second_term[...,None] * p1_s
        return first_term + second_term

    def vertex_voronai_to_unit_vertex_tf(self, vertex_index, theta, phi):
        vertices = self.vertices_tf
        vertex = vertices[vertex_index]
        boundary = self.verts_voronai.boundaries_tf[vertex_index]
        relative_boundary_theta = tf.linspace(0., 2*np.pi, len(boundary))
        spherical_edge_point = boundary[tf.math.argmin(tf.math.abs(relative_boundary_theta - theta))]
        edge_point = self.spherical_to_cartesian_tf([spherical_edge_point])[0]
        new_vertex = vertex + phi*(edge_point - vertex)
        unit_vertex = new_vertex / tf_safe_norm(new_vertex)
        return unit_vertex
    def voronai_vertices_to_unit_vertices_tf(self, thetas, phis):
        thetas_shape, phis_shape = thetas.shape,phis.shape
        thetas %= 2*np.pi
        phis /= 2.

        slerp_theta_starts_mask = tf.cast(thetas[...,None] >= self.verts_voronai.boundary_dihedral_angles_tf[None,...], tf.float32)
        slerp_theta_starts_i = tf.argmax(slerp_theta_starts_mask * self.verts_voronai.boundary_dihedral_angles_tf[None,...], axis=-1)
        batch_dims = tf.broadcast_to(tf.range(slerp_theta_starts_i.shape[-1], dtype=tf.int64)[None,:], slerp_theta_starts_i.shape)
        slerp_theta_starts_vertices = tf.gather_nd(self.verts_voronai.boundary_points_tf, tf.stack((batch_dims, slerp_theta_starts_i), axis=-1))
        slerp_theta_starts_angles = tf.gather_nd(self.verts_voronai.boundary_dihedral_angles_tf, tf.stack((batch_dims, slerp_theta_starts_i), axis=-1))
        slerp_theta_stops_i = slerp_theta_starts_i + 1
        slerp_theta_stops_vertices = tf.gather_nd(self.verts_voronai.boundary_points_tf, tf.stack((batch_dims, slerp_theta_stops_i), axis=-1))
        slerp_theta_stops_angles = tf.gather_nd(self.verts_voronai.boundary_dihedral_angles_tf, tf.stack((batch_dims, slerp_theta_stops_i), axis=-1))
        slerp_theta_ts = (thetas - slerp_theta_starts_angles) / (slerp_theta_stops_angles - slerp_theta_starts_angles)

        # spherical_starts = self.cartesian_to_spherical_tf(slerp_theta_starts_vertices)
        # spherical_stops = self.cartesian_to_spherical_tf(slerp_theta_stops_vertices)
        edge_points = slerp_theta_starts_vertices + slerp_theta_ts[...,None]*(slerp_theta_stops_vertices - slerp_theta_starts_vertices)
        edge_points /= tf_safe_norm(edge_points)[...,None]
        # edge_points = self.spherical_to_cartesian_tf(spherical_edge_points)

        new_vertices = self.vertices_tf[None,...] + phis[...,None]*(edge_points - self.vertices_tf[None,...])
        unit_vertices = new_vertices / tf_safe_norm(new_vertices)[...,None]
        tf.debugging.assert_all_finite(unit_vertices,"vvtuvtf")
        return tf.reshape(unit_vertices, thetas_shape + (3,))