import os
import argparse
import numpy as np
import torch
import imageio
import trimesh
import pyrender
import smplx

# -----------------------------------------------------
# You need install the following libraries
# pip install smplx pyrender trimesh imageio imageio-ffmpeg numpy
# -----------------------------------------------------


# -----------------------------------------------------
# helpers
# -----------------------------------------------------

def create_bone_lines(joints, parent_ids):
    """
    joints: (J, 3) numpy, in world/camera coordinates
    parent_ids: list/array of length J, parent index or -1
    returns: vertices (2*E, 3), index pairs for each line
    """
    lines = []
    for j, p in enumerate(parent_ids):
        if p < 0:
            continue
        lines.append((p, j))

    # each edge -> 2 vertices (we'll use LineSegments)
    verts = []
    idxs = []
    idx = 0
    for (p, c) in lines:
        verts.append(joints[p])
        verts.append(joints[c])
        idxs.append([idx, idx + 1])
        idx += 2

    return np.array(verts, dtype=np.float32), np.array(idxs, dtype=np.int32)


def load_smplx_model(model_folder, gender="neutral", device="cpu"):
    model = smplx.create(
        model_folder,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True
    ).to(device)
    return model


def decompose_smplx_params(params_frame, n_betas=10):
    """
    Minimal example assuming params layout:
    [trans(3), global_orient(3), body_pose(21*3), betas(n_betas),
     left_hand_pose(15*3), right_hand_pose(15*3), jaw_pose(3),
     leye_pose(3), reye_pose(3), expression(10)]
    Adjust indexing as needed for your data.
    """
    idx = 0
    trans = params_frame[idx:idx+3]; idx += 3
    global_orient = params_frame[idx:idx+3]; idx += 3
    body_pose = params_frame[idx:idx+21*3]; idx += 21*3
    betas = params_frame[idx:idx+n_betas]; idx += n_betas
    left_hand_pose = params_frame[idx:idx+15*3]; idx += 15*3
    right_hand_pose = params_frame[idx:idx+15*3]; idx += 15*3
    jaw_pose = params_frame[idx:idx+3]; idx += 3
    leye_pose = params_frame[idx:idx+3]; idx += 3
    reye_pose = params_frame[idx:idx+3]; idx += 3
    expression = params_frame[idx:idx+10]

    return dict(
        transl=trans,
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        expression=expression
    )


def smplx_forward(model, params_np, device="cpu"):
    params = decompose_smplx_params(params_np)
    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    output = model(
        transl=to_t(params["transl"]),
        global_orient=to_t(params["global_orient"]).view(1, 1, 3),
        body_pose=to_t(params["body_pose"]).view(1, 21, 3),
        betas=to_t(params["betas"]),
        left_hand_pose=to_t(params["left_hand_pose"]).view(1, 15, 3),
        right_hand_pose=to_t(params["right_hand_pose"]).view(1, 15, 3),
        jaw_pose=to_t(params["jaw_pose"]).view(1, 1, 3),
        leye_pose=to_t(params["leye_pose"]).view(1, 1, 3),
        reye_pose=to_t(params["reye_pose"]).view(1, 1, 3),
        expression=to_t(params["expression"])
    )
    verts = output.vertices[0].detach().cpu().numpy()       # (V, 3)
    joints = output.joints[0].detach().cpu().numpy()        # (J, 3)
    return verts, joints


def render_frame(verts, faces, joints, parent_ids,
                 scene, mesh_node, line_node, renderer,
                 img_size=(720, 720)):
    # update mesh
    mesh_trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    scene.remove_node(mesh_node)
    mesh_node = scene.add(mesh, name="smplx_mesh")

    # update bones
    bone_verts, bone_idx = create_bone_lines(joints, parent_ids)
    line_color = np.array([[0.0, 0.0, 0.0, 1.0]] * len(bone_verts))
    line = pyrender.Mesh.from_lines(bone_verts, bone_idx, colors=line_color)

    scene.remove_node(line_node)
    line_node = scene.add(line, name="skeleton")

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # convert RGBA->RGB
    color = color[:, :, :3]
    return color, mesh_node, line_node


# -----------------------------------------------------
# main
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_npy", type=str, required=True,
                        help="Path to SMPL-X motion .npy (T x D) file.")
    parser.add_argument("--smplx_model_folder", type=str, required=True,
                        help="Folder containing SMPL-X model files.")
    parser.add_argument("--gender", type=str, default="neutral",
                        choices=["neutral", "male", "female"])
    parser.add_argument("--output_mp4", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    motion = np.load(args.motion_npy)  # shape: (T, D)
    T = motion.shape[0]

    model = load_smplx_model(args.smplx_model_folder, gender=args.gender, device=device)
    faces = model.faces.astype(np.int32)

    # Skeleton parent indices for visualization
    # This should be consistent with SMPL-X joint regressor.
    # minimal example using first 22 body joints chain:
    parent_ids = np.array([
        -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9,  # etc...
    ])
    # If your output has more / different joints, adapt this.

    # Setup scene / renderer
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4, 1.0])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(camera, pose=np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.8],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ]))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, parent_node=cam_node)

    # dummy mesh & skeleton to init nodes
    dummy_verts = np.zeros((faces.max() + 1, 3), dtype=np.float32)
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(dummy_verts, faces, process=False))
    mesh_node = scene.add(mesh, name="smplx_mesh")

    dummy_line = pyrender.Mesh.from_lines(
        np.zeros((2, 3), dtype=np.float32),
        np.array([[0, 1]], dtype=np.int32)
    )
    line_node = scene.add(dummy_line, name="skeleton")

    renderer = pyrender.OffscreenRenderer(viewport_width=args.width,
                                          viewport_height=args.height)

    writer = imageio.get_writer(args.output_mp4, fps=args.fps)

    for t in range(T):
        params_t = motion[t]
        verts, joints = smplx_forward(model, params_t, device=device)

        # Optionally center / scale scene
        joints_center = joints[0]  # pelvis
        verts = verts - joints_center
        joints = joints - joints_center

        frame_img, mesh_node, line_node = render_frame(
            verts, faces, joints, parent_ids,
            scene, mesh_node, line_node, renderer,
            img_size=(args.height, args.width)
        )

        writer.append_data(frame_img.astype(np.uint8))
        print(f"Rendered frame {t+1}/{T}", end="\r")

    writer.close()
    renderer.delete()
    print(f"\nSaved video to {args.output_mp4}")


if __name__ == "__main__":
    main()


