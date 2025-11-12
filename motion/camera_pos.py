import numpy as np
import pyrender

def _ensure_viewer_like_camera(scene):
    """
    - 若 scene.main_camera_node 已存在，直接返回它；
    - 否则：基于场景中非地板几何自动求 ROI，按 Viewer 的风格布置相机，
      并在相机节点下附加 Raymond 三点灯光（与 viewer(use_raymond_lighting=True) 近似）。
    返回：相机节点（pyrender.Node）
    """
    # 已有主相机则直接用
    cam_node = getattr(scene, "main_camera_node", None)
    if cam_node is not None:
        return cam_node

    # ---- 1) 估计关注区域：忽略“地板/超薄大面”，优先人体/轨迹/物体/导航网格 ----
    def _node_world_pose(n):
        try:
            return scene.get_pose(n)  # world matrix
        except Exception:
            return getattr(n, "matrix", np.eye(4, dtype=np.float32))

    def _node_bounds_world(n):
        """从 primitive.positions 估计节点的世界 AABB（若取不到则返回 None）"""
        if getattr(n, "mesh", None) is None:
            return None
        ps = []
        for p in getattr(n.mesh, "primitives", []):
            v = getattr(p, "positions", None)
            if v is not None:
                ps.append(np.asarray(v, dtype=np.float32))
        if not ps:
            # 有些版本 Mesh 自带 bounds
            b = getattr(n.mesh, "bounds", None)
            if b is None:
                return None
            T = _node_world_pose(n)
            corners = np.array([[b[0,0], b[0,1], b[0,2], 1],
                                [b[1,0], b[1,1], b[1,2], 1]], dtype=np.float32) @ T.T
            bbmin = corners.min(0)[:3]; bbmax = corners.max(0)[:3]
            return bbmin, bbmax
        V = np.concatenate(ps, axis=0)
        T = _node_world_pose(n)
        Vh = np.c_[V, np.ones((V.shape[0], 1), dtype=np.float32)]
        VW = (Vh @ T.T)[:, :3]
        return VW.min(0), VW.max(0)

    # 选择候选节点：排除 floor / plane 等、优先包含 body/start/target/middle/navmesh/object/markers
    preferred_keys = ("body", "start", "target", "middle", "navmesh", "object", "goal", "marker")
    roi_mins, roi_maxs = [], []
    for n in scene.get_nodes():
        if getattr(n, "mesh", None) is None:
            continue
        name = (n.name or "").lower()
        # 明确跳过地板
        if "floor" in name:
            continue
        # 计算 AABB
        bb = _node_bounds_world(n)
        if bb is None:
            continue
        bmin, bmax = bb
        ext = bmax - bmin
        # 过滤“超薄大面”（很可能是地面或墙）——薄且 XY 很大
        is_thin_large = (ext[2] < 0.03) and (max(ext[0], ext[1]) > 2.0*min(1.0, ext[2]+1e-6))
        if is_thin_large and "navmesh" not in name:
            continue
        # 如果名字里包含优先关键字，优先保留
        weight = 2.0 if any(k in name for k in preferred_keys) else 1.0
        for _ in range(int(weight)):
            roi_mins.append(bmin)
            roi_maxs.append(bmax)

    if roi_mins:
        bmin = np.vstack(roi_mins).min(0)
        bmax = np.vstack(roi_maxs).max(0)
        center = 0.5 * (bmin + bmax)
        extent = (bmax - bmin)
        # 用平面半径估计覆盖范围，让整段轨迹落在视野内
        radius = 0.5 * float(max(extent[0], extent[1]))
        # 留一点边框
        radius = max(radius, 0.5) * 1.15
    else:
        # 兜底：用 scene 的质心与尺度（与 Viewer 一致的思路）
        center = np.asarray(scene.centroid, dtype=np.float32)
        scale = scene.scale if scene.scale > 0 else 2.0
        radius = max(scale, 1.0)

    # ---- 2) 构建相机：yfov=pi/3，按半径反推距离；斜上方观察轨迹平面 ----
    yfov = np.pi / 3.0  # 与 Viewer 默认一致
    dist = radius / np.tan(yfov / 2.0)
    # 机位：从“斜上方”看向中心，适合看运动轨迹
    eye = center + np.array([dist/np.sqrt(2), -dist/np.sqrt(2), dist*0.5], dtype=np.float32)
    # eye = center + np.array([dist/np.sqrt(2), dist*0.5, dist/np.sqrt(2)], dtype=np.float32)
    target = np.array([center[0], center[1], center[2]], dtype=np.float32)

    def _look_at(eye, target, up=(0, 1, 0)):
        eye, target, up = map(lambda a: np.asarray(a, dtype=np.float32), (eye, target, up))
        z = eye - target; z /= (np.linalg.norm(z) + 1e-8)
        x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-8)
        y = np.cross(z, x)
        T = np.eye(4, dtype=np.float32); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
        return T

    cam = pyrender.PerspectiveCamera(yfov=yfov)
    cam_pose = _look_at(eye, target, up=(0, 0, 1))

    cam_node = pyrender.Node(camera=cam, matrix=cam_pose)
    scene.add_node(cam_node)
    scene.main_camera_node = cam_node  # 与 Viewer 的做法一致

    # ---- 3) 在相机下附加 Raymond 三点灯（与 viewer(use_raymond_lighting=True) 近似）----
    scene.ambient_light = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
    thetas = np.pi * np.array([1/6, 1/6, 1/6], dtype=np.float32)
    phis   = np.pi * np.array([0.0, 2/3, 4/3], dtype=np.float32)
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi); yp = np.sin(theta) * np.sin(phi); zp = np.cos(theta)
        z = np.array([xp, yp, zp], dtype=np.float32); z /= (np.linalg.norm(z) + 1e-8)
        x = np.array([-z[1], z[0], 0.0], dtype=np.float32)
        if np.linalg.norm(x) < 1e-8: x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x /= (np.linalg.norm(x) + 1e-8); y = np.cross(z, x)
        Lpose = np.eye(4, dtype=np.float32); Lpose[:3, :3] = np.c_[x, y, z]
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0/3.0)
        scene.add_node(pyrender.Node(light=light, matrix=Lpose), parent_node=cam_node)

    return cam_node
