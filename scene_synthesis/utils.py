
import numpy as np
from PIL import Image
import trimesh

from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
import seaborn as sns

def get_textured_objects(bbox_params_t, objects_dataset, classes, diffusion=False, no_texture=False, render_bboxes=False):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    model_jids = []
    if diffusion:
        start, end = 0, bbox_params_t.shape[1]
    else:
        #for autoregressive model, we delete the 'start' and 'end'
        start, end = 1, bbox_params_t.shape[1]-1

    color_palette = np.array(sns.color_palette('hls', len(classes)-2))

    for j in range(start, end):
        query_size = bbox_params_t[0, j, -4:-1]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )

        # Load the furniture and scale it as it is given in the dataset
        if no_texture:
            class_index = bbox_params_t[0, j, :-7].argmax(-1)
            raw_mesh = Mesh.from_file(furniture.raw_model_path, color=color_palette[class_index, :])
        else:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        
        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        if no_texture:
            color = color_palette[class_index, :]
            tr_mesh.visual.vertex_colors = (color[None, :].repeat(tr_mesh.vertices.shape[0], axis=0).reshape(-1, 3) * 255.0).astype(np.uint8)
            tr_mesh.visual.face_colors = (color[None, :].repeat(tr_mesh.faces.shape[0], axis=0).reshape(-1, 3) * 255.0).astype(np.uint8)
        else:
            tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
            tr_mesh.visual.vertex_colors = (tr_mesh.visual.to_color()).vertex_colors[:, 0:3]
            print('convert texture to vertex colors')
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)
        model_jids.append( (furniture.raw_model_path).split('/')[-2] )

    return renderables, trimesh_meshes, model_jids


def get_textured_objects_based_on_objfeats(bbox_params_t, objects_dataset, classes, diffusion=False, no_texture=False, query_objfeats=None, combine_size=False, render_bboxes=False):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    model_jids = []

    if diffusion:
        start, end = 0, bbox_params_t.shape[1]
    else:
        #for autoregressive model, we delete the 'start' and 'end'
        start, end = 1, bbox_params_t.shape[1]-1

    color_palette = np.array(sns.color_palette('hls', len(classes)-2))

    for j in range(start, end):
        query_size = bbox_params_t[0, j, -4:-1]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        if combine_size:
            furniture = objects_dataset.get_closest_furniture_to_objfeats_and_size(
                query_label, query_objfeats[0, j], query_size
            )
        else:
            furniture = objects_dataset.get_closest_furniture_to_objfeats(
                query_label, query_objfeats[0, j]
            )

        # Load the furniture and scale it as it is given in the dataset
        if no_texture:
            class_index = bbox_params_t[0, j, :-7].argmax(-1)
            raw_mesh = Mesh.from_file(furniture.raw_model_path, color=color_palette[class_index, :])
        else:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        
        # instead of using retrieved object scale, we use predicted size
        raw_bbox_vertices = np.load(furniture.path_to_bbox_vertices, mmap_mode="r") #np.array(raw_mesh.bounding_box.vertices)
        raw_sizes = np.array([
            np.sqrt(np.sum((raw_bbox_vertices[4]-raw_bbox_vertices[0])**2))/2,
            np.sqrt(np.sum((raw_bbox_vertices[2]-raw_bbox_vertices[0])**2))/2,
            np.sqrt(np.sum((raw_bbox_vertices[1]-raw_bbox_vertices[0])**2))/2
        ])
        raw_mesh.scale(query_size / raw_sizes)
        #print('raw mesh sizes is {}, and the desired size is {}, the computed scale is {}'.format(raw_sizes, query_size, query_size/raw_sizes))

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        if no_texture:
            color=color_palette[class_index, :]
            tr_mesh.visual.vertex_colors = (color[None, :].repeat(tr_mesh.vertices.shape[0], axis=0).reshape(-1, 3) * 255.0).astype(np.uint8)
            tr_mesh.visual.face_colors = (color[None, :].repeat(tr_mesh.faces.shape[0], axis=0).reshape(-1, 3) * 255.0).astype(np.uint8)
        else:
            tr_mesh.visual.material.image = Image.open(
                furniture.texture_image_path
            )
            tr_mesh.visual.vertex_colors = (tr_mesh.visual.to_color()).vertex_colors[:, 0:3]
        # tr_mesh.vertices *= furniture.scale
        # use the calculated scale from query size and retrieved object size :
        tr_mesh.vertices *= (query_size / raw_sizes)
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)
        model_jids.append( (furniture.raw_model_path).split('/')[-2] )

    return renderables, trimesh_meshes, model_jids



def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )

    return floor, tr_floor
