from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


ROOT = Path(r"D:\Desktop\Glove Tracking\code")
CLIP_ID = "clip-000001"
CLIP_TAR = ROOT / "data" / "train_quest3" / f"{CLIP_ID}.tar"
PROCESSED_CLIP_DIR = ROOT / "data" / "train_quest3_processed" / CLIP_ID

IMPORT_LEFT_GLOVE = True
IMPORT_RIGHT_GLOVE = False
IMPORT_LEFT_HAND = False
IMPORT_RIGHT_HAND = False
IMPORT_CAMERA1 = False
IMPORT_CAMERA2 = True

START_FRAME = 1
FRAME_STEP = 1
FPS = 30
INTERPOLATION = "CONSTANT"
UPRIGHT_ROLL_DEG = -90.0
BLENDER_CAMERA_LOCAL_ROLL_DEG = 180.0
EXPORT_ABC = False
ABC_PATH = PROCESSED_CLIP_DIR / "sequences_3d" / "hot3d_import.abc"

CANONICAL_CAMERA = {
    "name": "canonical_rgb_c1",
    "width": 1280,
    "height": 720,
    "fx": 563.5018310546875,
    "fy": 563.4933471679688,
    "cx": 642.6881713867188,
    "cy": 355.161865234375,
}

HAND_MATERIAL_COLOR = (0.042, 0.042, 0.042, 1.0)  # #3A3A3AFF


def selected_stream_ids() -> tuple[str, ...]:
    stream_ids: list[str] = []
    if IMPORT_CAMERA1:
        stream_ids.append("1201-1")
    if IMPORT_CAMERA2:
        stream_ids.append("1201-2")
    return tuple(stream_ids)


def glove_sequence_dir(hand: str) -> Path:
    return PROCESSED_CLIP_DIR / "sequences_3d" / "glove" / hand


def mano_sequence_dir(hand: str) -> Path:
    return PROCESSED_CLIP_DIR / "sequences_3d" / "mano" / hand


def glove_file_glob(hand: str) -> str:
    return f"glove_{hand}_frame*.obj"


def mano_file_glob(hand: str) -> str:
    return f"mano_{hand}_frame*.obj"


def glove_object_name(hand: str) -> str:
    return f"GloveShellSequence_{hand}_{CLIP_ID}"


def mano_object_name(hand: str) -> str:
    return f"MANOSequence_{hand}_{CLIP_ID}"


def hot3d_world_to_blender_world_basis() -> Matrix:
    return Matrix(
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def rotation_z_homogeneous(degrees: float) -> Matrix:
    return Matrix.Rotation(math.radians(degrees), 4, "Z")


def parse_obj_geometry(
    path: Path, apply_basis: bool = True
) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    basis = hot3d_world_to_blender_world_basis()
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                v = Vector((float(x), float(y), float(z), 1.0))
                if apply_basis:
                    v = basis @ v
                vertices.append((v.x, v.y, v.z))
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for part in parts:
                    face.append(int(part.split("/")[0]) - 1)
                if len(face) >= 3:
                    faces.append(tuple(face))
    return vertices, faces


def load_obj_vertices_only(path: Path) -> list[float]:
    basis = hot3d_world_to_blender_world_basis()
    coords: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                v = basis @ Vector((float(x), float(y), float(z), 1.0))
                coords.extend((v.x, v.y, v.z))
    return coords


def create_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return obj


def import_base_obj_with_materials(path: Path, object_name: str) -> bpy.types.Object:
    existing_names = set(bpy.data.objects.keys())
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))

    imported = [
        obj
        for obj in bpy.data.objects
        if obj.name not in existing_names and obj.type == "MESH"
    ]
    if not imported:
        raise RuntimeError(f"Failed to import OBJ: {path}")
    if len(imported) > 1:
        raise RuntimeError(
            f"Expected a single mesh from {path}, got {len(imported)} objects"
        )
    obj = imported[0]
    obj.name = object_name
    obj.data.name = f"{object_name}Mesh"
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return obj


def apply_basis_to_mesh_data(obj: bpy.types.Object) -> None:
    basis = hot3d_world_to_blender_world_basis()
    obj.data.transform(basis)
    obj.data.update()


def delete_object_if_exists(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def delete_mesh_if_exists(name: str) -> None:
    mesh = bpy.data.meshes.get(name)
    if mesh is not None:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def cleanup_existing_sequence_objects() -> None:
    object_names = [
        glove_object_name("left"),
        glove_object_name("right"),
        mano_object_name("left"),
        mano_object_name("right"),
    ]
    for obj_name in object_names:
        delete_object_if_exists(obj_name)
        delete_mesh_if_exists(f"{obj_name}Mesh")


def cleanup_existing_cameras() -> None:
    for stream_id in ("1201-1", "1201-2"):
        delete_object_if_exists(f"HOT3D_{stream_id}")
        delete_object_if_exists(f"HOT3D_{stream_id}_Rig")
        cam_data = bpy.data.cameras.get(f"HOT3D_{stream_id}")
        if cam_data is not None:
            bpy.data.cameras.remove(cam_data, do_unlink=True)


def set_keyframe_interpolation(obj) -> None:
    anim = getattr(obj, "animation_data", None)
    if anim is None or anim.action is None:
        return
    fcurves = getattr(anim.action, "fcurves", None)
    if fcurves is None:
        return
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = INTERPOLATION


def set_shape_key_interpolation(obj) -> None:
    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        return
    anim = shape_keys.animation_data
    if anim is None or anim.action is None:
        return
    fcurves = getattr(anim.action, "fcurves", None)
    if fcurves is None:
        return
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = INTERPOLATION


def ensure_material(
    obj: bpy.types.Object,
    name: str,
    base_color: tuple[float, float, float, float],
    alpha: float = 1.0,
    roughness: float = 0.55,
    metallic: float = 0.0,
) -> None:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = base_color
        principled.inputs["Alpha"].default_value = alpha
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED" if alpha < 1.0 else "OPAQUE"
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def build_sequence_animation(
    sequence_dir: Path,
    file_glob: str,
    object_name: str,
    import_materials: bool = True,
    override_material: dict | None = None,
) -> bpy.types.Object:
    obj_paths = sorted(sequence_dir.glob(file_glob))
    if not obj_paths:
        raise FileNotFoundError(f"No OBJ files matched {file_glob} in {sequence_dir}")

    scene = bpy.context.scene
    scene.render.fps = FPS

    if import_materials:
        base_obj = import_base_obj_with_materials(obj_paths[0], object_name)
    else:
        vertices, faces = parse_obj_geometry(obj_paths[0], apply_basis=False)
        base_obj = create_mesh_object(object_name, vertices, faces)
    apply_basis_to_mesh_data(base_obj)
    if override_material is not None:
        ensure_material(
            base_obj,
            name=override_material["name"],
            base_color=override_material["base_color"],
            alpha=override_material.get("alpha", 1.0),
            roughness=override_material.get("roughness", 0.55),
            metallic=override_material.get("metallic", 0.0),
        )

    if base_obj.data.shape_keys is None:
        base_obj.shape_key_add(name="Basis")
    base_obj.data.shape_keys.use_relative = True

    shape_key_names = []
    for frame_idx, obj_path in enumerate(obj_paths):
        coords = load_obj_vertices_only(obj_path)
        if len(coords) != len(base_obj.data.vertices) * 3:
            raise ValueError(
                f"Vertex count mismatch at {obj_path.name}: "
                f"{len(coords) // 3} vs {len(base_obj.data.vertices)}"
            )

        if frame_idx > 0:
            key_block = base_obj.shape_key_add(name=obj_path.stem, from_mix=False)
            key_block.data.foreach_set("co", coords)
            key_block.value = 0.0
            shape_key_names.append(key_block.name)

    for frame_idx in range(len(obj_paths)):
        scene_frame = START_FRAME + frame_idx * FRAME_STEP
        for key_name in shape_key_names:
            key_block = base_obj.data.shape_keys.key_blocks[key_name]
            key_block.value = 0.0
            key_block.keyframe_insert(data_path="value", frame=scene_frame)
        if frame_idx > 0:
            active_key = base_obj.data.shape_keys.key_blocks[shape_key_names[frame_idx - 1]]
            active_key.value = 1.0
            active_key.keyframe_insert(data_path="value", frame=scene_frame)

    scene.frame_start = START_FRAME
    scene.frame_end = START_FRAME + (len(obj_paths) - 1) * FRAME_STEP
    set_shape_key_interpolation(base_obj)
    return base_obj


def build_glove_animation(hand: str) -> bpy.types.Object:
    return build_sequence_animation(
        sequence_dir=glove_sequence_dir(hand),
        file_glob=glove_file_glob(hand),
        object_name=glove_object_name(hand),
        import_materials=True,
        override_material=None,
    )


def build_mano_animation(hand: str) -> bpy.types.Object:
    return build_sequence_animation(
        sequence_dir=mano_sequence_dir(hand),
        file_glob=mano_file_glob(hand),
        object_name=mano_object_name(hand),
        import_materials=True,
        override_material={
            "name": f"MANOCompareMaterial_{hand}",
            "base_color": HAND_MATERIAL_COLOR,
            "alpha": 1.0,
            "roughness": 1.0,
            "metallic": 0.0,
        },
    )


def hot3d_quat_trans_to_matrix(quat_wxyz, trans_xyz) -> Matrix:
    quat = Quaternion((quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    mat = quat.to_matrix().to_4x4()
    mat.translation = Vector(trans_xyz)
    return mat


def blender_camera_local_to_hot3d_camera_local_basis() -> Matrix:
    return Matrix(
        (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def canonical_hot3d_camera_world(T_world_from_camera: Matrix) -> Matrix:
    return T_world_from_camera @ rotation_z_homogeneous(UPRIGHT_ROLL_DEG)


def hot3d_to_blender_camera_world(T_world_from_camera: Matrix) -> Matrix:
    basis_world = hot3d_world_to_blender_world_basis()
    basis_cam_local = blender_camera_local_to_hot3d_camera_local_basis()
    return basis_world @ T_world_from_camera @ basis_cam_local


def create_or_get_camera_rig(stream_id: str) -> bpy.types.Object:
    rig_name = f"HOT3D_{stream_id}_Rig"
    rig_obj = bpy.data.objects.get(rig_name)
    if rig_obj is None:
        rig_obj = bpy.data.objects.new(rig_name, None)
        rig_obj.empty_display_type = "PLAIN_AXES"
        bpy.context.scene.collection.objects.link(rig_obj)
    rig_obj.rotation_mode = "QUATERNION"
    return rig_obj


def create_or_get_camera(stream_id: str) -> bpy.types.Object:
    obj_name = f"HOT3D_{stream_id}"
    cam_data = bpy.data.cameras.get(obj_name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(obj_name)
    cam_obj = bpy.data.objects.get(obj_name)
    if cam_obj is None:
        cam_obj = bpy.data.objects.new(obj_name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.rotation_mode = "QUATERNION"
    rig_obj = create_or_get_camera_rig(stream_id)
    if cam_obj.parent != rig_obj:
        cam_obj.parent = rig_obj

    width = CANONICAL_CAMERA["width"]
    height = CANONICAL_CAMERA["height"]
    fx = float(CANONICAL_CAMERA["fx"])
    cx = float(CANONICAL_CAMERA["cx"])
    cy = float(CANONICAL_CAMERA["cy"])

    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = cam_data.sensor_width * height / width
    cam_data.lens = fx / width * cam_data.sensor_width
    cam_data.shift_x = -(cx - width * 0.5) / width
    cam_data.shift_y = (cy - height * 0.5) / width
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100.0
    cam_obj.location = (0.0, 0.0, 0.0)
    cam_obj.rotation_mode = "XYZ"
    cam_obj.rotation_euler = (0.0, 0.0, math.radians(BLENDER_CAMERA_LOCAL_ROLL_DEG))
    cam_obj.rotation_mode = "QUATERNION"
    return cam_obj


def animate_cameras() -> None:
    stream_ids = selected_stream_ids()
    if not stream_ids:
        return

    with tarfile.open(CLIP_TAR, "r") as tar:
        frame_keys = sorted(
            name.split(".info.json")[0] for name in tar.getnames() if name.endswith(".info.json")
        )
        bpy.context.scene.render.resolution_x = int(CANONICAL_CAMERA["width"])
        bpy.context.scene.render.resolution_y = int(CANONICAL_CAMERA["height"])
        primary_camera = None
        for frame_idx, frame_key in enumerate(frame_keys):
            scene_frame = START_FRAME + frame_idx * FRAME_STEP
            cameras = json.load(tar.extractfile(f"{frame_key}.cameras.json"))
            for stream_id in stream_ids:
                camera_entry = cameras[stream_id]
                cam_obj = create_or_get_camera(stream_id)
                if primary_camera is None or stream_id == "1201-2":
                    primary_camera = cam_obj
                rig_obj = create_or_get_camera_rig(stream_id)
                source_world_mat = hot3d_quat_trans_to_matrix(
                    camera_entry["T_world_from_camera"]["quaternion_wxyz"],
                    camera_entry["T_world_from_camera"]["translation_xyz"],
                )
                canonical_world_mat = canonical_hot3d_camera_world(source_world_mat)
                rig_obj.matrix_world = hot3d_to_blender_camera_world(canonical_world_mat)
                rig_obj.keyframe_insert(data_path="location", frame=scene_frame)
                rig_obj.keyframe_insert(data_path="rotation_quaternion", frame=scene_frame)
                set_keyframe_interpolation(rig_obj)
        if primary_camera is not None:
            bpy.context.scene.camera = primary_camera


def export_alembic() -> None:
    bpy.ops.wm.alembic_export(
        filepath=str(ABC_PATH),
        start=bpy.context.scene.frame_start,
        end=bpy.context.scene.frame_end,
        xsamples=1,
        gsamples=1,
        flatten=False,
    )


def main() -> None:
    scene = bpy.context.scene
    scene.render.fps = FPS

    cleanup_existing_sequence_objects()
    cleanup_existing_cameras()

    imported_assets: list[str] = []
    if IMPORT_LEFT_GLOVE:
        build_glove_animation("left")
        imported_assets.append("left_glove")
    if IMPORT_RIGHT_GLOVE:
        build_glove_animation("right")
        imported_assets.append("right_glove")
    if IMPORT_LEFT_HAND:
        build_mano_animation("left")
        imported_assets.append("left_hand")
    if IMPORT_RIGHT_HAND:
        build_mano_animation("right")
        imported_assets.append("right_hand")

    animate_cameras()

    if EXPORT_ABC:
        export_alembic()
        print(f"Exported Alembic to {ABC_PATH}")

    print(
        f"Imported clip {CLIP_ID}: assets={imported_assets}, "
        f"camera1={IMPORT_CAMERA1}, camera2={IMPORT_CAMERA2}."
    )


if __name__ == "__main__":
    main()
