import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize, least_squares, dual_annealing
from skimage.morphology import ball
from stardist import relabel_image_patchdist, Rays_Patch, relabel_image_stardist3D, Rays_GoldenSpiral
from stardist.bezier_utils import get_control_points, get_vertex_normals, subdivide_tri_tf_bary_one_shot
from stardist.matching import matching_dataset
from tifffile import imwrite
import napari
import seaborn as sns

sns.set_theme()

ray_counts = [6, 12, 24, 48, 96]
radii = [5, 10, 25, 50, 75]
patch_ious = {}
star_ious = {}

# radius = 5
# edge_dist, face_dist = 5.277102278165789, 5.0451362012542225
# vertex_normals = get_vertex_normals(patch_rays.vertices, patch_rays.faces, patch_rays.vertextofacemap)
# face_vertices = patch_rays.vertices[patch_rays.faces]
# face_normals = vertex_normals[patch_rays.faces]
# control_points = get_control_points(face_vertices, face_normals)
# control_points[:,:3,:] *= radius
# control_points[:,3:9,:] *= edge_dist
# control_points[:,9,:] *= face_dist
# addl_subdivided_vertices = subdivide_tri_tf_bary_one_shot(tf.constant(control_points, dtype=tf.float32), patch_rays, 4)
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_points(5*patch_rays.vertices+5, size=0.1, face_color='red')
# viewer.add_points(addl_subdivided_vertices.numpy()+5, size=0.1, face_color='green')
# viewer.add_points(control_points.reshape((-1,3))+5, size=0.1, face_color='blue')
# viewer.add_image(ball(5))
# napari.run()

for n_rays in ray_counts:
    print("n_rays",n_rays)
    patch_ious[n_rays] = []
    star_ious[n_rays] = []
    for radius in radii:
        Y = ball(radius)

        if n_rays != 96:
            patch_rays = Rays_Patch(n_rays, anisotropy=None)
            def get_sub_norms_mean(input_tuple):
                edge_dist, face_dist = input_tuple
                vertex_normals = get_vertex_normals(patch_rays.vertices, patch_rays.faces, patch_rays.vertextofacemap)
                face_vertices = patch_rays.vertices[patch_rays.faces]
                face_normals = vertex_normals[patch_rays.faces]
                control_points = get_control_points(face_vertices, face_normals)
                control_points[:,:3,:] *= radius+1
                control_points[:,3:9,:] *= edge_dist
                control_points[:,9,:] *= face_dist
                addl_subdivided_vertices = subdivide_tri_tf_bary_one_shot(tf.constant(control_points, dtype=tf.float32), patch_rays, 4)
                return radius+1 - np.linalg.norm(addl_subdivided_vertices, axis=-1)

            minimized = least_squares(get_sub_norms_mean, [radius+1, radius+1], method='trf')
            edge_dist, face_dist = minimized.x
            opt = minimized.fun
            # print("e, d", edge_dist, face_dist, opt, radius)

            dist = np.concatenate((np.ones(len(patch_rays))*radius, np.ones(2*len(patch_rays.edges))*edge_dist, np.ones(len(patch_rays.faces))*face_dist))
            Y_reconstructed_patch = [relabel_image_patchdist(Y, dist, patch_rays)]
            iou_patch = matching_dataset([Y], Y_reconstructed_patch, thresh=0, show_progress=False).mean_true_score
            patch_ious[n_rays].append(iou_patch)

        star_rays = Rays_GoldenSpiral(n_rays, anisotropy=None)
        Y_reconstructed_star = [relabel_image_stardist3D(Y, star_rays)]
        iou_star = matching_dataset([Y], Y_reconstructed_star, thresh=0, show_progress=False).mean_true_score
        star_ious[n_rays].append(iou_star)

# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(Y)
# viewer.add_image(Y_reconstructed_patch)
# viewer.add_image(Y_reconstructed_star)
# napari.run()


for n_rays in ray_counts[:-1]:
    plt.plot(radii, patch_ious[n_rays], 'o-', label='SurfDist ({} rays)'.format(n_rays))
for n_rays in ray_counts:
    plt.plot(radii, star_ious[n_rays], 'o--', label='StarDist-3D ({} rays)'.format(n_rays))
plt.xlabel('Radius (voxels)')
plt.ylabel('Reconstruction score (intersection over union)')
plt.title('Reconstruction of spherical input')
plt.legend()
plt.show()