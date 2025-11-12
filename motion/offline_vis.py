import smplx
import pdb
import torch
import pickle
import trimesh
import tqdm
import pyrender
from pyrender import Node, PerspectiveCamera, DirectionalLight
import numpy as np
import time
import glob
import argparse
import os
from copy import deepcopy
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
from camera_pos import _ensure_viewer_like_camera
jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

unity_to_zup = np.array(
            [[-1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )

shapenet_to_zup = np.array(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )
def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}
def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}

# === [新增]：让 offscreen 的相机与 Viewer 一致 ===
def _attach_viewer_raymond_lights(scene, cam_node, lighting_intensity=3.0):
    """复刻 Viewer 的 Raymond 三灯并挂到相机节点下。"""
    # 与 Viewer 中 _create_raymond_lights 的极角/方位角相同  :contentReference[oaicite:11]{index=11}
    thetas = np.pi * np.array([1/6, 1/6, 1/6])
    phis   = np.pi * np.array([0.0, 2.0/3.0, 4.0/3.0])

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4); T[:3, :3] = np.c_[x, y, z]

        # Viewer 渲染时把 3 盏灯的强度设为 vli/3，各灯挂在相机节点下面  :contentReference[oaicite:12]{index=12}
        light = DirectionalLight(color=np.ones(3), intensity=lighting_intensity / 3.0)
        scene.add_node(Node(light=light, matrix=T), parent_node=cam_node)

from scipy.spatial.transform import Rotation
def rollout_primitives(motion_primitives):
    smplx_param_list = []
    gender = motion_primitives[0]['gender']
    body_model = smplx.create(model_path=model_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=20,
                              ).to(device='cuda')
    for idx, motion_primitive in enumerate(motion_primitives):
        pelvis_original = body_model(betas=torch.cuda.FloatTensor(motion_primitive['betas']).repeat(20, 1)).joints[:, 0, :].detach().cpu().numpy()  # [10, 3]
        smplx_param = motion_primitive['smplx_params'][0]  #[10, 96]

        rotation = motion_primitive['transf_rotmat'].reshape((3, 3)) # [3, 3]
        transl = motion_primitive['transf_transl'].reshape((1, 3)) # [1, 3]
        smplx_param[:, :3] = np.matmul((smplx_param[:, :3] + pelvis_original), rotation.T) - pelvis_original + transl
        r_ori = Rotation.from_rotvec(smplx_param[:, 3:6])
        r_new = Rotation.from_matrix(np.tile(motion_primitive['transf_rotmat'], [20, 1, 1])) * r_ori
        smplx_param[:, 3:6] = r_new.as_rotvec()

        if idx == 0:
            start_frame = 0
        elif motion_primitive['mp_type'] == '1-frame':
            start_frame = 1
        elif motion_primitive['mp_type'] == '2-frame':
            start_frame = 2
        else:
            print(motion_primitive['mp_type'])
            # crowd-env use 1 frame model at the moment
            start_frame = 1
        smplx_param = smplx_param[start_frame:, :]
        smplx_param_list.append(smplx_param)


    return  np.concatenate(smplx_param_list, axis=0)  # [t, 96]

def vis_results_new(result_paths, vis_marker=False, vis_pelvis=True, vis_object=False,
                vis_navmesh=True, start_frame=0,
                slow_rate=1, save_path=None, add_floor=True):
    scene = pyrender.Scene()

    # [修改] —— 导入 imageio，并创建离屏渲染器与帧缓存（仅在需要保存时启用）
    import imageio  # [修改]
    # viewport = (1280, 720)  # [修改] 你可按需调分辨率
    viewport = (640, 360)
    offscreen = pyrender.OffscreenRenderer(viewport_width=viewport[0], viewport_height=viewport[1]) if save_path is not None else None  # [修改]
    _frames = []  # [修改]

    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    scene.add_node(axis_node)
    # assume all have same wpath
    motions_list = []
    wpath = []
    wpath_orients = None
    target_orient = None
    object_mesh = None
    floor_height = 0
    for input_path in result_paths:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            motions = data['motion']
            gender = motions[0]['gender']
            motions_list.append(motions)
            wpath.append(data['wpath'] if vis_pelvis else None)
            wpath_orients = data['wpath_orients'] if 'wpath_orients' in data else None
            target_orient = data['target_orient'] if 'target_orient' in data else None
            floor_height = data['floor_height'] if 'floor_height' in data else 0
            box = data['box'] if 'box' in data else None
            if vis_marker:
                if 'goal_markers' in data: # only target markers
                    markers = data['goal_markers']
                elif 'markers' in data: # start and target markers
                    markers = data['markers'].reshape(-1, 3)
                else:
                    markers = None
            else:
                markers = None
            if vis_object and object_mesh is None:
                if 'obj_path' in data:
                    y_up_to_z_up = np.eye(4)
                    y_up_to_z_up[:3, :3] = np.array(
                        [[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]]
                    )
                    object_mesh = trimesh.load(
                        data['obj_path']
                    )
                    object_mesh.apply_transform(y_up_to_z_up)
                elif 'scene_path' in data:
                    obj_path = data['scene_path']
                    if 'exploration' in str(obj_path):
                        obj_path = '/home/genli/Desktop/cvpr/gamma_interaction/data/scenes/test_exploration/7.80_6.34_1_1700921858.7392697.ply' 
                    object_mesh = trimesh.load(obj_path)
                    if 'obj_transform' in data:
                        object_mesh.apply_transform(data['obj_transform'])
                    if 'floor_height' in data:
                        object_mesh.vertices[:, 2] -= data['floor_height']
                else:
                    obj_id = data['obj_id']
                    transform = data['obj_transform']
                    if type(transform) == torch.Tensor:
                        transform = transform.detach().cpu().numpy()
                    if type(transform) == tuple:
                        transform = transform[0]
                    object_mesh = trimesh.load(
                        os.path.join(*([shapenet_dir] + (obj_id.split('-') if '-' in obj_id else obj_id.split('_')) + ['models', 'model_normalized.obj'])),
                        force='mesh'
                    )
                    object_mesh.apply_transform(transform)

                m = pyrender.Mesh.from_trimesh(object_mesh)
                object_node = pyrender.Node(mesh=m, name='object')
                scene.add_node(object_node)
                if 'navmesh_path' in data and vis_navmesh:
                    navmesh = trimesh.load(data['navmesh_path'], force='mesh')
                    if 'obj' in str(data['navmesh_path']):
                        navmesh.apply_transform(unity_to_zup)
                    else:
                        pass
                    navmesh.visual.vertex_colors = np.array([0, 0, 200, 20])
                    navmesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(navmesh), name='navmesh')
                    scene.add_node(navmesh_node)

    if add_floor:
        floor = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                     transform=np.array([[1.0, 0.0, 0.0, 0],
                                                         [0.0, 1.0, 0.0, 0],
                                                         [0.0, 0.0, 1.0, floor_height-0.005],
                                                         [0.0, 0.0, 0.0, 1.0],
                                                         ]),)
        floor.visual.vertex_colors = [0.8, 0.8, 0.8]
        floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
        scene.add_node(floor_node)

    body_model = smplx.create(model_path=model_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=len(motions_list),
                              ).to(device)
    pelvis_nodes = []
    if vis_pelvis:
        for idx, ww in enumerate(wpath): 
            sm = trimesh.creation.uv_sphere(radius=0.05)
            sm.visual.vertex_colors = np.asarray(scalarMap.to_rgba(idx / len(wpath))[:3]) * 255
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[:, :3, 3] = ww[0]
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            start_node = pyrender.Node(mesh=m, name='start')
            sm = trimesh.creation.uv_sphere(radius=0.05)
            sm.visual.vertex_colors = np.asarray(scalarMap.to_rgba(idx / len(wpath))[:3]) * 255
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[:, :3, 3] = ww[-1]
            m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            target_node = pyrender.Node(mesh=m, name='target')
            pelvis_nodes = [start_node, target_node]
            if len(ww) > 2:
                sm = trimesh.creation.uv_sphere(radius=0.05)
                sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                tfs = np.tile(np.eye(4), (len(ww) - 2, 1, 1))
                tfs[:, :3, 3] = ww[1:-1]
                m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                middle_node = pyrender.Node(mesh=m, name='middle')
                pelvis_nodes.append(middle_node)
            if wpath_orients is not None:
                from scipy.spatial.transform import Rotation as R
                for point_idx in range(len(wpath_orients)):
                    trans_mat = np.eye(4)
                    trans_mat[:3, 3] = ww[point_idx]
                    trans_mat[:3, :3] = R.from_rotvec(wpath_orients[point_idx]).as_matrix()
                    point_axis = trimesh.creation.axis(transform=trans_mat)
                    pelvis_nodes.append(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(point_axis, smooth=False)))
            if target_orient is not None:
                from scipy.spatial.transform import Rotation as R
                trans_mat = np.eye(4)
                trans_mat[:3, 3] = ww[-1]
                trans_mat[:3, :3] = R.from_rotvec(target_orient).as_matrix()
                point_axis = trimesh.creation.axis(transform=trans_mat)
                pelvis_nodes.append(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(point_axis, smooth=False)))

            for pelvis_node in pelvis_nodes:
                scene.add_node(pelvis_node)

    if vis_marker and markers is not None:
        sm = trimesh.creation.uv_sphere(radius=0.02)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(markers), 1, 1))
        tfs[:, :3, 3] = markers
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        marker_node = pyrender.Node(mesh=m, name='goal_markers')
        scene.add_node(marker_node)

    if box is not None:
        extents = box[1] - box[0]
        transform = np.eye(4)
        transform[:3, 3] = 0.5 * (box[0] + box[1])
        box_mesh = trimesh.creation.box(extents=extents,
                             transform=transform,
                                        vertex_color=(255, 255, 0))
        m = pyrender.Mesh.from_trimesh(box_mesh)
        box_node = pyrender.Node(mesh=m, name='box')
        scene.add_node(box_node)
        
    # === [新增调用]：相机 & 灯光与 Viewer 保持一致 ===
    cam_node = scene.main_camera_node or _ensure_viewer_like_camera(scene)
    _attach_viewer_raymond_lights(scene, cam_node, lighting_intensity=3.0)

    num_sequences = len(motions_list)
    rollout_frames_list = [rollout_primitives(motions) for motions in motions_list]
    print(np.array([len(frames) for frames in rollout_frames_list]))
    max_frame = np.array([len(frames) for frames in rollout_frames_list]).max()

    rollout_frames_pad_list = []  # [T_max, 93], pad shorter sequences with last frame
    for idx in range(len(rollout_frames_list)):
        frames = rollout_frames_list[idx]
        rollout_frames_pad_list.append(np.concatenate([frames, np.tile(frames[-1:, :], (max_frame + 1 - frames.shape[0], 1))], axis=0))
    smplx_params = np.stack(rollout_frames_pad_list, axis=0)  # [S, T_max, 93]
    betas = [motions[0]['betas'] for motions in motions_list]
    betas = np.stack(betas, axis=0)  # [S, 10]
    body_node = None
    transls = []
    global_orients = []
    body_poses = []
    for frame_idx in tqdm.tqdm(range(start_frame, max_frame)):
        transls.append(smplx_params[:, frame_idx, :3])
        global_orients.append(smplx_params[:, frame_idx, 3:6])
        body_poses.append(smplx_params[:, frame_idx, 6:69])
        smplx_dict = {
            'betas': betas,
            'transl': smplx_params[:, frame_idx, :3],
            'global_orient': smplx_params[:, frame_idx, 3:6],
            'body_pose': smplx_params[:, frame_idx, 6:69],
        }
        smplx_dict = params2torch(smplx_dict)

        output = body_model(**smplx_dict)
        vertices = output.vertices.detach().cpu().numpy()
        body_meshes = []
        for seq_idx in range(vertices.shape[0]):
            m = trimesh.Trimesh(vertices=vertices[seq_idx], faces=body_model.faces, process=False)
            m.visual.vertex_colors[:, 3] = 160
            m.visual.vertex_colors[:, :3] = np.asarray(scalarMap.to_rgba(seq_idx / vertices.shape[0])[:3]) * 255
            body_meshes.append(m)
        body_mesh = pyrender.Mesh.from_trimesh(body_meshes, smooth=False)
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=body_mesh, name='body')
        scene.add_node(body_node)

        # [修改] —— 在每帧更新后进行离屏抓帧
        if offscreen is not None:  # [修改]
            color, _ = offscreen.render(scene)  # [修改]
            _frames.append(color.copy())       # [修改]

        time.sleep(0.025 * slow_rate)

    print("max depth: ", len(motions_list[0]))
    dist = 0.
    speed = []
    for kk in range(len(transls) - 1):
        dist += np.linalg.norm(transls[kk+1] - transls[kk])
        speed.append(np.linalg.norm(transls[kk+1] - transls[kk]) / 0.025)
    print("walking dist : ", dist)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # [新] 将文件名从 .gif 更改为 .mp4
        out_mp4 = os.path.join(save_path, f"{time.time():.0f}.mp4")
        if len(_frames) > 0:
            # [新] 使用 'libx264' 编码器保存为 MP4
            # 'pixelformat='yuv420p' 确保与大多数播放器（如 Windows Media Player, QuickTime）兼容
            try:
                imageio.mimsave(
                    out_mp4, 
                    _frames, 
                    fps=30, 
                    codec='libx264',
                    pixelformat='yuv420p'
                )
                print(f"Saved MP4 to: {out_mp4}")
            except Exception as e:
                print(f"保存 MP4 失败: {e}")
                print("错误：无法保存 MP4。这通常是因为缺少 'ffmpeg' 库。")
                print("请尝试使用 'pip install imageio[ffmpeg]' 或 'conda install -c conda-forge ffmpeg' 来安装它。")

        else:
            print("警告：未抓到任何帧（_frames 为空），请检查循环是否执行或 offscreen.render 是否被调用。")
        
        if offscreen is not None:
            offscreen.delete()
            
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--env', type=str, default='0')
parser.add_argument('--slow_rate', type=int, default=1)
parser.add_argument('--path', type=str, default='')
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--max_vis', type=int, default=8, help="maximum number of sequences to be visualized")
args = parser.parse_args()

model_path = "data/smplx/models"
shapenet_dir = '/mnt/atlas_root/vlg-data/ShapeNetCore.v2/'
device = torch.device('cuda')

vis_results_new(
    list(glob.glob(args.path))[:args.max_vis],
    start_frame=args.start_frame,
    vis_navmesh=False,
    vis_marker=True, vis_pelvis=True, vis_object=True, add_floor=True,
    slow_rate=args.slow_rate,
    save_path='./visual'
)
# python offline_vis.py --path ./log/mid_eval_results/motion_1762869044.7913988.pkl
