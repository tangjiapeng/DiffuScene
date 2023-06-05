

import os
from turtle import color

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44

import trimesh

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects, get_textured_objects_based_on_objfeats
import torch
import open3d as o3d

class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid
    # Return a simple-3dviz renderable
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,
    path_to_floor_plan_textures,
    without_room_mask=False,
    no_texture=False,
):
    if not without_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_mask = None
    # Also get a renderable for the floor plan
    if no_texture:
        floor, tr_floor = get_floor_plan_white(scene)
    else:
        floor, tr_floor = get_floor_plan(
            scene,
            [
                os.path.join(path_to_floor_plan_textures, fi)
                for fi in os.listdir(path_to_floor_plan_textures)
            ]
        )
    return [floor], [tr_floor], room_mask


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


def get_textured_objects_in_scene(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes:
        model_path = furniture.raw_model_path
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = TexturedMesh.from_file(model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
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
    return renderables


## render floor with white background , not texture
def get_floor_plan_white(scene):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz Mesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    floor = Mesh.from_faces(
        vertices=vertices,
        faces=faces,
        colors=np.ones((len(vertices), 3))*[1.0, 1.0, 1.0],
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )

    # trimesh.visual.face_colors
    tr_floor.visual.vertex_colors = (np.ones((len(vertices), 3)) * 255).astype(np.uint8)

    return floor, tr_floor


def get_colored_objects_in_scene(scene, colors, ignore_lamps=False):
    renderables = []
    for furniture, c in zip(scene.bboxes, colors):
        model_path = furniture.raw_model_path
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = Mesh.from_file(model_path, color=c)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
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
    return renderables



def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes, names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            m,
            return_texture=True
        )

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].faces).shape[0]
        num_vertex_colors += np.asarray(meshes[i].visual.vertex_colors/255.0).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].faces)
        current_vertex_colors = np.asarray(meshes[i].visual.vertex_colors/255.0)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors[:, 0:3] 

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:-1].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def render_to_folder(
    args,
    folder,
    filename,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    scene_top2down,
    bbox_params,
    add_start_end=False,
    diffusion=False,
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    if args.retrive_objfeats:
        objfeats = boxes["objfeats"].cpu().numpy()
        print('shape retrieval based on obj latent feats')
        decoded_surf = None
        renderables, trimesh_meshes, model_jids = get_textured_objects_based_on_objfeats(
            bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels), diffusion=diffusion, no_texture=args.no_texture,
            query_objfeats=objfeats, query_surfs=decoded_surf, 
        )
    else:
        renderables, trimesh_meshes, model_jids = get_textured_objects(
            bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels), diffusion=diffusion,  no_texture=args.no_texture,
        )

    if not args.without_floor:
        renderables += floor_plan
        trimesh_meshes += tr_floor
        
    
    if args.save_mesh:
        path_to_objs = os.path.join(args.output_directory, folder)
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
            
        if args.no_texture:
            # save whole scene as a single mesh
            path_to_scene = os.path.join(args.output_directory, folder, filename+args.mesh_format)
            whole_scene_mesh = merge_meshes( trimesh_meshes )
            o3d.io.write_triangle_mesh(path_to_scene, whole_scene_mesh)
        else:
            path_to_scene = os.path.join(path_to_objs, filename)
            if not os.path.exists(path_to_scene):
                os.mkdir(path_to_scene)
            export_scene(path_to_scene, trimesh_meshes)

    
    path_to_image_folder = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_image_folder):
        os.mkdir(path_to_image_folder)
    path_to_image = os.path.join(args.output_directory, folder, filename + "_render.png")
    
    if args.render_top2down:
        # render_top2down is render:
        render(
            scene_top2down,
            renderables,
            color=None,
            mode="shading",
            frame_path=path_to_image,
        )
    else:
        behaviours = [
            LightToCamera(),
            SaveFrames(path_to_image, 1)
        ]
        render_simple_3dviz(
            renderables,
            behaviours=behaviours,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=args.n_frames,
            scene=scene
        )


def render_scene_from_bbox_params(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    scene_top2down,
    path_to_image,
    path_to_objs,
    filename,
    network_objae=None,
    device=None,
    diffusion=False,
):
    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()
    
    if args.retrive_objfeats:
        objfeats = boxes["objfeats"].cpu().numpy()
        if args.retrive_surface:
            print('shape retrieval based on decoded surface')
            B, N, C = boxes["objfeats"].shape
            with torch.no_grad():
                decoded_surf = network_objae.decode(boxes["objfeats"].to(device).float().reshape(B*N, C))
                assert decoded_surf.shape[-1] == 3
                decoded_surf = decoded_surf.reshape(B, N, -1, 3)
                decoded_surf = decoded_surf.cpu().numpy()
        else:
            print('shape retrieval based on obj latent feats')
            decoded_surf = None

        renderables, trimesh_meshes, model_jids = get_textured_objects_based_on_objfeats(
            bbox_params_t, objects_dataset, classes, diffusion=diffusion, no_texture=args.no_texture,
            query_objfeats=objfeats, query_surfs=decoded_surf, 
        )
    else:
        renderables, trimesh_meshes, model_jids = get_textured_objects(
            bbox_params_t, objects_dataset, classes, diffusion=diffusion, no_texture=args.no_texture,
        )
        
    if not args.without_floor:
        renderables += floor_plan
        trimesh_meshes += tr_floor

    # Do the rendering
    if args.render_top2down:
        #render_top2down is render:
        render(
            scene_top2down,
            renderables,
            color=None,
            mode="shading",
            frame_path=path_to_image,
        )
    else:
        behaviours = [
            LightToCamera(),
            SaveFrames(path_to_image+".png", 1)
        ]
        render_simple_3dviz(
            renderables,
            behaviours=behaviours,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=args.n_frames,
            scene=scene
        )
        
    if args.save_mesh:
        if trimesh_meshes is not None:
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            # Create a trimesh scene and export it
            if args.no_texture:
                path_to_scene = os.path.join(path_to_objs, filename+args.mesh_format)
                whole_scene_mesh = merge_meshes( trimesh_meshes )
                o3d.io.write_triangle_mesh(path_to_scene, whole_scene_mesh)
            else:
                path_to_scene = os.path.join(path_to_objs, filename)
                if not os.path.exists(path_to_scene):
                    os.mkdir(path_to_scene)
                export_scene(path_to_scene, trimesh_meshes)
