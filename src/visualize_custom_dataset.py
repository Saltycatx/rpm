import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from tqdm import tqdm

# =====================================================================
# 0. 强制 Matplotlib 使用纯离线无头后端 (彻底解决 X11 BadWindow 崩溃)
# =====================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 1. 强制本地导入 src/models (复刻 a.py 逻辑)
# =====================================================================
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_to_remove = [k for k in sys.modules if k == 'models' or k.startswith('models.')]
for k in _to_remove:
    del sys.modules[k]

import types
import importlib.util

def _load_local_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None: return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_models_dir = os.path.join(_SRC_DIR, 'models')
build_rpmnet_surgical = None
if os.path.exists(_models_dir):
    _models_pkg = types.ModuleType('models')
    _models_pkg.__path__ = [_models_dir]
    _models_pkg.__package__ = 'models'
    sys.modules['models'] = _models_pkg

    _load_local_module('models.pointnet_util', os.path.join(_models_dir, 'pointnet_util.py'))
    _load_local_module('models.feature_nets', os.path.join(_models_dir, 'feature_nets.py'))
    _rpmnet_mod = _load_local_module('learning3d.models.rpmnet', os.path.join(_models_dir, 'rpmnet.py'))
    if _rpmnet_mod:
        build_rpmnet_surgical = _rpmnet_mod.get_model

# =====================================================================
# 2. 导入 learning3d 基础模型 (用于对比测试)
# =====================================================================
try:
    from learning3d.models import DGCNN, DCP, PointNet, PointNetLK, iPCRNet, PPFNet, RPMNet as BaseRPMNet, PRNet
except ImportError:
    print("⚠️ 无法导入 learning3d 库，仅能测试 SurgicalRPMNet。")

from data_loader.surgical_registration_dataset import SurgicalRegistrationData


# =====================================================================
# 3. 数学工具与预处理 (复刻 a.py)
# =====================================================================
def get_transform_from_corres(P, Q):
    P_xyz = P[:, :3]
    Q_xyz = Q[:, :3]
    centroid_P = np.mean(P_xyz, axis=0)
    centroid_Q = np.mean(Q_xyz, axis=0)
    p = P_xyz - centroid_P
    q = Q_xyz - centroid_Q
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def denormalize_transform(T_norm, centroid, scale):
    """基于 a.py 改进版的反归一化公式"""
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] *= scale
    C = np.eye(4, dtype=np.float64)
    C[:3, 3] = centroid
    S_inv = np.eye(4, dtype=np.float64)
    S_inv[:3, :3] *= 1.0 / scale
    C_inv = np.eye(4, dtype=np.float64)
    C_inv[:3, 3] = -centroid
    return C @ S @ T_norm @ S_inv @ C_inv

def preprocess_dataset_sample(pts_t, pts_s, num_points=1024, normal_radius=0.1, normal_max_nn=30):
    pcd_t = o3d.geometry.PointCloud()
    pcd_s = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pts_t)
    pcd_s.points = o3d.utility.Vector3dVector(pts_s)

    pcd_t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    pcd_s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))

    pts_t_arr = np.asarray(pcd_t.points)
    nrm_t_arr = np.asarray(pcd_t.normals)
    pts_s_arr = np.asarray(pcd_s.points)
    nrm_s_arr = np.asarray(pcd_s.normals)

    # a.py 中的居中逻辑
    centroid_t = np.mean(pts_t_arr, axis=0)
    pts_t_c = pts_t_arr - centroid_t
    pts_s_c = pts_s_arr - centroid_t
    scale = np.max(np.linalg.norm(pts_t_c, axis=1))
    if scale < 1e-8:
        scale = 1.0
    pts_t_c /= scale
    pts_s_c /= scale

    def sample_data(pts, nrms, n):
        if pts.shape[0] >= n:
            idx = np.random.choice(pts.shape[0], n, replace=False)
        else:
            idx = np.random.choice(pts.shape[0], n, replace=True)
        return pts[idx], nrms[idx]

    pts_t_s, nrm_t_s = sample_data(pts_t_c, nrm_t_arr, num_points)
    pts_s_s, nrm_s_s = sample_data(pts_s_c, nrm_s_arr, num_points)

    t3 = torch.from_numpy(pts_t_s).float().unsqueeze(0)
    s3 = torch.from_numpy(pts_s_s).float().unsqueeze(0)
    t6 = torch.from_numpy(np.hstack([pts_t_s, nrm_t_s])).float().unsqueeze(0)
    s6 = torch.from_numpy(np.hstack([pts_s_s, nrm_s_s])).float().unsqueeze(0)

    return t3, s3, t6, s6, centroid_t, scale

def refine_registration_icp(source_pts, target_pts, initial_transform, distance_threshold=0.05, max_iter=200):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return reg.transformation


# =====================================================================
# 4. 模型加载与推理
# =====================================================================
def build_model(name, device):
    if name == 'SurgicalRPMNet':
        if build_rpmnet_surgical is None:
            raise RuntimeError("找不到本地 models/rpmnet.py，无法加载 SurgicalRPMNet！请确保从 src/ 目录下运行此脚本。")
        # 带上 TPF 组合的参数！
        model_args = argparse.Namespace(
            features=['ppf', 'dxyz', 'xyz', 'tpf'],
            feat_dim=96,
            radius=0.3,
            num_neighbors=64,
            no_slack=False,
            num_sk_iter=5,
        )
        return build_rpmnet_surgical(model_args)
    elif name == 'DCP': return DCP(feature_model=DGCNN(emb_dims=512), cycle=True)
    elif name == 'PointNetLK': return PointNetLK(feature_model=PointNet(emb_dims=1024, use_bn=True))
    elif name == 'PCRNet': return iPCRNet(feature_model=PointNet(emb_dims=1024))
    elif name == 'RPMNet': return BaseRPMNet(feature_model=PPFNet())
    elif name == 'PRNet': return PRNet(emb_dims=512, num_iters=3)
    return None

def load_model(name, path, device):
    model = build_model(name, device)
    if model is None or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()

def run_model_inference(model, model_name, t_in, s_in, device):
    # ==== SurgicalRPMNet 专用前向推理 ====
    if model_name == 'SurgicalRPMNet':
        data = {
            'points_ref': t_in.to(device),
            'points_src': s_in.to(device),
        }
        with torch.no_grad():
            pred_transforms, endpoints = model(data, 5) # num_reg_iter = 5
        T_34 = pred_transforms[-1][0].cpu().numpy().astype(np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = T_34
        return T

    # ==== learning3d 基础模型推理 ====
    source_np = s_in.cpu().numpy()[0, :3, :] if s_in.shape[1] == 3 else s_in.cpu().numpy()[0, :, :3]
    if s_in.shape[1] == 3: source_np = source_np.T

    with torch.no_grad():
        if model_name == 'PRNet':
            dummy_R = torch.eye(3).unsqueeze(0).to(device)
            dummy_t = torch.zeros(1, 3).to(device)
            output = model(t_in, s_in, dummy_R, dummy_t)
            R_ab = output['est_R'][0].cpu().numpy()
            t_ba = output['est_t'][0].cpu().numpy()
            t_ab = -np.dot(R_ab, t_ba)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3], T[:3, 3] = R_ab, t_ab
            return T

        output = model(t_in, s_in)
        if isinstance(output, dict):
            if 'R' in output and 't' in output:
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = output['R'][0].cpu().numpy()
                T[:3, 3] = output['t'][0].cpu().numpy()
                return T
            elif 'est_R' in output and 'est_t' in output:
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = output['est_R'][0].cpu().numpy()
                T[:3, 3] = output['est_t'][0].cpu().numpy()
                return T
            elif 'transformation' in output:
                return output['transformation'][0].cpu().numpy().astype(np.float64)
    return np.eye(4, dtype=np.float64)


# =====================================================================
# 5. 可视化工具 (单例模式解决 EGL 崩溃)
# =====================================================================
_global_renderer = None

def get_gradient_transforms(num_steps=10, max_angle=90.0, max_trans=1.0):
    transforms = []
    angles = np.linspace(max_angle / num_steps, max_angle, num_steps)
    translations = np.linspace(max_trans / num_steps, max_trans, num_steps)
    for ang, trans in zip(angles, translations):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.radians(ang))
        t = np.random.randn(3)
        t = (t / np.linalg.norm(t)) * trans
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        transforms.append((T, ang, trans))
    return transforms

def plot_registration_results(model_name, template_pts, results, filename="visualization.png"):
    global _global_renderer
    n = len(results)
    
    views = [
        (np.array([ 1.0,  1.0,  1.0]), np.array([0.0, 0.0, 1.0]), "Isometric View"),
        (np.array([ 0.0,  0.0,  1.0]), np.array([0.0, 1.0, 0.0]), "Top View"),
        (np.array([ 1.0,  0.0,  0.0]), np.array([0.0, 0.0, 1.0]), "Front View"),
        (np.array([ 0.0,  1.0,  0.0]), np.array([0.0, 0.0, 1.0]), "Side View")
    ]
    
    fig = plt.figure(figsize=(4 * n, 16))
    fig.suptitle(f"{model_name} + ICP Registration\n(Red: Template, Blue: Probe Path)", fontsize=24, y=0.98)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    
    if _global_renderer is None:
        _global_renderer = o3d.visualization.rendering.OffscreenRenderer(600, 600)
    
    render = _global_renderer
    render.scene.set_background([1.0, 1.0, 1.0, 1.0]) 

    center = template_pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(template_pts - center, axis=1))
    cam_distance = max_dist * 2.0 

    for row_idx, (eye_offset, up, view_name) in enumerate(views):
        eye_dir = eye_offset / np.linalg.norm(eye_offset)
        eye = center + eye_dir * cam_distance

        for col_idx, (src_aligned, ang, trans) in enumerate(results):
            t_pcd = o3d.geometry.PointCloud()
            t_pcd.points = o3d.utility.Vector3dVector(template_pts)
            t_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            
            s_pcd = o3d.geometry.PointCloud()
            s_pcd.points = o3d.utility.Vector3dVector(src_aligned)
            s_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            
            render.scene.add_geometry("template", t_pcd, mat)
            render.scene.add_geometry("source", s_pcd, mat)
            
            render.setup_camera(60.0, center, eye, up)
            img = np.asarray(render.render_to_image())
            
            render.scene.remove_geometry("template")
            render.scene.remove_geometry("source")

            ax = fig.add_subplot(4, n, row_idx * n + col_idx + 1)
            ax.imshow(img)
            ax.axis('off')
            
            if row_idx == 0:
                ax.set_title(f"Sample {col_idx+1}\nAng: {ang:.1f}°, Trans: {trans:.2f}m", fontsize=14)
            ax.text(0.05, 0.95, view_name, transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存 {model_name} 的可视化结果至: {filename}")


# =====================================================================
# 6. 主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_dcp', type=str, default='learning3d/pretrained/exp_dcp/models/best_model.t7')
    parser.add_argument('--w_pnlk', type=str, default='learning3d/pretrained/exp_pnlk/models/best_model.t7')
    parser.add_argument('--w_pcrnet', type=str, default='learning3d/pretrained/exp_ipcrnet/models/best_model.t7')
    parser.add_argument('--w_prnet', type=str, default='learning3d/pretrained/exp_prnet/models/best_model.t7')
    parser.add_argument('--w_rpmnet', type=str, default='learning3d/pretrained/exp_rpmnet/models/partial-trained.pth')
    parser.add_argument('--w_srpmnet', type=str, default='./model-best1.pth', help='填入真实的训练权重路径')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=5, help='测试样本数（列数）')
    parser.add_argument('--icp_threshold', type=float, default=0.05)
    parser.add_argument('--icp_max_iter', type=int, default=200)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading un-transformed base template and probe path from dataset...")
    dataset = SurgicalRegistrationData(
        data_root="./datasets/modelnet40_ply_hdf5_2048", # 注意检查你的数据集路径
        partition='test', 
        template_points=2048,
        source_points=2000,
        angle_range=0,
        translation_range=0.0,
        noise_sigma=0.005,
        coverage_ratio=0.90,
    )
    
    sample = dataset[1]
    if isinstance(sample, dict):
        base_template_np = sample['points_ref'][:, :3] 
        base_probe_np = sample['points_src'][:, :3] 
    else:
        base_template_np = sample[0].numpy()
        base_probe_np = sample[1].numpy()

    gradient_transforms = get_gradient_transforms(num_steps=args.num_samples, max_angle=90.0, max_trans=1.0)

    tasks = [
        # ('DCP', args.w_dcp),
        # ('PointNetLK', args.w_pnlk),
        # ('PCRNet', args.w_pcrnet),
        # ('PRNet', args.w_prnet),
        # ('RPMNet', args.w_rpmnet),
        ('SurgicalRPMNet', args.w_srpmnet),
    ]

    for model_name, weight_path in tasks:
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {model_name}, 权重不存在: {weight_path}")
            continue

        print(f"\n⚙️ 正在使用 {model_name} 进行预测并生成可视化...")
        model = load_model(model_name, weight_path, device)
        if model is None:
            continue

        results_for_plot = []

        for T_gt, ang, trans in tqdm(gradient_transforms, desc=f"{model_name} Inference"):
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(base_probe_np)
            pcd_source.transform(T_gt)
            source_np = np.asarray(pcd_source.points)

            t3, s3, t6, s6, centroid_t, scale = preprocess_dataset_sample(
                base_template_np, source_np, num_points=1024
            )
            
            # 特征维度分发
            if model_name in ['RPMNet', 'SurgicalRPMNet']:
                t_in, s_in = t6.to(device), s6.to(device)
            else:
                t_in, s_in = t3.to(device), s3.to(device)

            # DL 初始推理
            T_DL_norm = run_model_inference(model, model_name, t_in, s_in, device)
            T_DL_real = denormalize_transform(T_DL_norm, centroid_t, scale)

            # ICP 优化
            T_Final_real = refine_registration_icp(
                source_pts=source_np, 
                target_pts=base_template_np, 
                initial_transform=T_DL_real,
                distance_threshold=args.icp_threshold,
                max_iter=args.icp_max_iter
            )

            # 对齐源点云
            pcd_aligned = o3d.geometry.PointCloud()
            pcd_aligned.points = o3d.utility.Vector3dVector(source_np)
            pcd_aligned.transform(T_Final_real)
            aligned_np = np.asarray(pcd_aligned.points)

            results_for_plot.append((aligned_np, ang, trans))

        filename = f"vis_{model_name.lower()}_with_icp.png"
        plot_registration_results(model_name, base_template_np, results_for_plot, filename=filename)
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()