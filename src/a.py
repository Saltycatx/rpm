"""Evaluate trained RPMNet/RPMNetSurgical on the SurgicalRegistrationData.

Usage (run from src/ directory):
    python eval_surgical.py \
        --method rpmnet_surgical \
        --dataset_type surgical \
        --dataset_path ../datasets/modelnet40_ply_hdf5_2048 \
        --resume ../logs/xxx/ckpt/model-best.pth \
        --features ppf dxyz xyz tpf \
        --num_reg_iter 5 \
        --num_test_samples 1000 \
        --gpu 0
"""
import os
import sys

# ====== 强制本地导入，彻底避免 site-packages 冲突 ======
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# 清除所有可能被 site-packages 缓存的 models 相关模块
_to_remove = [k for k in sys.modules if k == 'models' or k.startswith('models.')]
for k in _to_remove:
    del sys.modules[k]

# 用 runpy 的思路，直接 exec 加载本地 rpmnet.py
import types
import importlib.util

def _load_local_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# 按依赖顺序加载：pointnet_util -> feature_nets -> rpmnet
_models_dir = os.path.join(_SRC_DIR, 'models')

# 先注册 models 这个 namespace
_models_pkg = types.ModuleType('models')
_models_pkg.__path__ = [_models_dir]
_models_pkg.__package__ = 'models'
sys.modules['models'] = _models_pkg

_load_local_module('models.pointnet_util', os.path.join(_models_dir, 'pointnet_util.py'))
_load_local_module('models.feature_nets', os.path.join(_models_dir, 'feature_nets.py'))
_rpmnet_mod = _load_local_module('models.rpmnet', os.path.join(_models_dir, 'rpmnet.py'))

"""测试 copilot/rpmnet-architecture-innovation 分支训练的 RPMNet / RPMNetSurgical

保持和 test_custom_dataset.py 完全一致的评测流程：
  - 逐样本推理（不走 DataLoader batch）
  - 归一化 → 推理 → 反归一化 → RRE/RTE
  - 支持 DL-only 与 DL+ICP 两阶段报告
  - 支持 1000+ 样本（replacement 采样）

关键区别：用仓库自带 src/models/rpmnet.py 的 RPMNetEarlyFusion，而非 learning3d。

用法 (从 src/ 目录运行):
    python test_rpmnet_surgical.py \
        --resume ../logs/xxx/ckpt/model-best.pth \
        --num_samples 1000 \
        --angle_range 45 \
        --gpu 0
"""
import os
import sys
import copy
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from tqdm import tqdm

# =====================================================================
# 确保 src/ 目录在 path 中，以便 import models / common / data_loader
# =====================================================================
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.rpmnet import get_model as build_rpmnet
from common.torch import to_numpy, CheckPointManager
from data_loader.surgical_registration_dataset import SurgicalRegistrationData


# =====================================================================
# 1. 数学工具 (同 test_custom_dataset.py)
# =====================================================================
def get_transform_from_corres(P, Q):
    """根据对应点集计算 SVD 刚体变换, P->Q"""
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


def compute_registration_error(T_gt, T_pred):
    """计算 RRE (度) 和 RTE (m)"""
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]

    R_diff = R_gt.T @ R_pred
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    rre = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi
    rte = np.linalg.norm(t_gt - t_pred)
    return rre, rte


def denormalize_transform(T_norm, centroid, scale):
    """把归一化空间变换还原到真实坐标"""
    S = np.eye(4, dtype=np.float64)
    S[:3, :3] *= scale

    C = np.eye(4, dtype=np.float64)
    C[:3, 3] = centroid

    S_inv = np.eye(4, dtype=np.float64)
    S_inv[:3, :3] *= 1.0 / scale

    C_inv = np.eye(4, dtype=np.float64)
    C_inv[:3, 3] = -centroid

    T_real = C @ S @ T_norm @ S_inv @ C_inv
    return T_real


def preprocess_dataset_sample(pts_t, pts_s, num_points=1024, normal_radius=0.1, normal_max_nn=30):
    """
    输出:
      t3,s3: [1,N,3] 归一化xyz
      t6,s6: [1,N,6] 归一化xyz + normals
      centroid_t, scale: 反归一化参数
    """
    pcd_t = o3d.geometry.PointCloud()
    pcd_s = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pts_t)
    pcd_s.points = o3d.utility.Vector3dVector(pts_s)

    pcd_t.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    pcd_s.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )

    pts_t_arr = np.asarray(pcd_t.points)
    nrm_t_arr = np.asarray(pcd_t.normals)
    pts_s_arr = np.asarray(pcd_s.points)
    nrm_s_arr = np.asarray(pcd_s.normals)

    # 以 template 为中心归一化
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

    t3 = torch.from_numpy(pts_t_s).float().unsqueeze(0)       # [1,N,3]
    s3 = torch.from_numpy(pts_s_s).float().unsqueeze(0)
    t6 = torch.from_numpy(np.hstack([pts_t_s, nrm_t_s])).float().unsqueeze(0)  # [1,N,6]
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
# 2. 模型加载（直接用仓库的 RPMNetEarlyFusion）
# =====================================================================
def make_model_args(args):
    """把命令行参数组装成 get_model() 需要的 namespace"""
    model_args = argparse.Namespace(
        features=args.features,
        feat_dim=args.feat_dim,
        radius=args.radius,
        num_neighbors=args.num_neighbors,
        no_slack=args.no_slack,
        num_sk_iter=args.num_sk_iter,
    )
    return model_args


def load_model(args, device):
    model_args = make_model_args(args)
    model = build_rpmnet(model_args)

    # 加载 checkpoint
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"❌ 权重不存在: {args.resume}")

    checkpoint = torch.load(args.resume, map_location='cpu')
    # 兼容多种格式
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    print(f"✅ 加载模型: {args.resume}")
    return model


# =====================================================================
# 3. 推理：用仓库 RPMNet 的 forward(data_dict, num_iter)
# =====================================================================
def run_model_inference(model, ref_6d, src_6d, device, num_reg_iter=5):
    """
    ref_6d: [1, N, 6]  (归一化xyz + normals)
    src_6d: [1, N, 6]
    返回归一化空间下 T_norm (4x4 np.float64), source -> reference
    """
    # 构造 RPMNet 需要的 data dict
    data = {
        'points_ref': ref_6d.to(device),
        'points_src': src_6d.to(device),
    }

    with torch.no_grad():
        # model.forward 返回 (transforms_list, endpoints)
        # transforms_list 是 list of (B, 3, 4) tensors
        pred_transforms, endpoints = model(data, num_reg_iter)

    # 取最后一次迭代的变换
    T_34 = pred_transforms[-1][0].detach().cpu().numpy().astype(np.float64)  # (3,4)

    T_norm = np.eye(4, dtype=np.float64)
    T_norm[:3, :] = T_34
    return T_norm


# =====================================================================
# 4. 指标计算 (同 test_custom_dataset.py)
# =====================================================================
def calculate_metrics(errors_list):
    errors = np.array(errors_list)
    rre_arr, rte_arr = errors[:, 0], errors[:, 1]

    rre_mse = np.mean(rre_arr ** 2)
    rre_mae = np.mean(np.abs(rre_arr))
    rte_mse = np.mean(rte_arr ** 2)
    rte_mae = np.mean(np.abs(rte_arr))

    # 成功标准（可调）
    success_mask = (rte_arr < 0.01) & (rre_arr < 1.0)
    success_rate = np.mean(success_mask) * 100.0

    return rre_mse, rre_mae, rte_mse, rte_mae, success_rate


# =====================================================================
# 5. 主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='测试 RPMNet Surgical (仓库模型)')

    # 权重
    parser.add_argument('--resume', type=str, required=True,
                        help='训练好的 checkpoint 路径 (.pth)')

    # 样本数
    parser.add_argument('--num_samples', type=int, default=1000)

    # ICP 参数
    parser.add_argument('--icp_threshold', type=float, default=0.05)
    parser.add_argument('--icp_max_iter', type=int, default=200)
    parser.add_argument('--no_icp', action='store_true', help='跳过 ICP 精修阶段')

    # 设备
    parser.add_argument('--gpu', type=int, default=0)

    # 数据参数
    parser.add_argument('--dataset_path', type=str,
                        default='./datasets/modelnet40_ply_hdf5_2048')
    parser.add_argument('--template_points', type=int, default=2048)
    parser.add_argument('--source_points', type=int, default=2000)
    parser.add_argument('--angle_range', type=float, default=45.0)
    parser.add_argument('--translation_range', type=float, default=1.0)
    parser.add_argument('--noise_sigma', type=float, default=0.005)
    parser.add_argument('--coverage_ratio', type=float, default=0.90)

    # 预处理参数
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--normal_radius', type=float, default=0.1)
    parser.add_argument('--normal_max_nn', type=int, default=30)

    # 模型参数 (必须和训练时一致！)
    parser.add_argument('--features', nargs='+', type=str,
                        default=['ppf', 'dxyz', 'xyz'],
                        help="训练时的 features，如需 tpf 请加上: --features ppf dxyz xyz tpf")
    parser.add_argument('--feat_dim', type=int, default=96)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--num_neighbors', type=int, default=64)
    parser.add_argument('--no_slack', action='store_true')
    parser.add_argument('--num_sk_iter', type=int, default=5)
    parser.add_argument('--num_reg_iter', type=int, default=5)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 设备 ──
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── 数据集 ──
    print("Loading dataset...")
    dataset = SurgicalRegistrationData(
        data_root=args.dataset_path,
        partition='test',
        template_points=args.template_points,
        source_points=args.source_points,
        angle_range=args.angle_range,
        translation_range=args.translation_range,
        noise_sigma=args.noise_sigma,
        coverage_ratio=args.coverage_ratio,
    )
    total = len(dataset)
    num_samples = min(args.num_samples, total) if args.num_samples <= total else args.num_samples
    if num_samples > total:
        indices = np.random.choice(total, num_samples, replace=True).tolist()
    else:
        indices = np.random.choice(total, num_samples, replace=False).tolist()
    print(f"数据集大小: {total}, 评测样本数: {num_samples}")

    # ── 模型 ──
    model = load_model(args, device)

    # ── 推理 ──
    geo_errors = []
    icp_errors = []
    total_time = 0.0

    for idx in tqdm(indices, desc="Testing RPMNet", ncols=80):
        sample = dataset[idx]

        # SurgicalRegistrationData.__getitem__ 返回 dict:
        #   points_src (N,6), points_ref (M,6), transform_gt (3,4), points_raw (M,6)
        # 但如果你的数据集返回 (template_t, source_t, igt_t) 三元组:
        if isinstance(sample, dict):
            # copilot/rpmnet-architecture-innovation 分支的 dataset 返回 dict
            pts_ref = sample['points_ref']  # numpy (M, 6)
            pts_src = sample['points_src']  # numpy (N, 6)

            # 真实坐标
            pts_t_real = pts_ref[:, :3] if pts_ref.shape[1] >= 3 else pts_ref
            pts_s_real = pts_src[:, :3] if pts_src.shape[1] >= 3 else pts_src

            # GT transform: (3,4) → (4,4)
            gt_34 = sample['transform_gt']  # numpy (3,4)
            T_gt = np.eye(4, dtype=np.float64)
            T_gt[:3, :] = gt_34.astype(np.float64)
        else:
            # test_custom_dataset.py 风格: (template_t, source_t, igt_t)
            template_t, source_t, igt_t = sample
            pts_t_real = template_t.numpy() if torch.is_tensor(template_t) else template_t
            pts_s_real = source_t.numpy() if torch.is_tensor(source_t) else source_t
            igt_np = igt_t.numpy() if torch.is_tensor(igt_t) else igt_t
            T_gt = np.linalg.inv(igt_np) if igt_np.shape == (4, 4) else igt_np

        # 归一化 + 法向量估计
        t3, s3, t6, s6, centroid_t, scale = preprocess_dataset_sample(
            pts_t_real, pts_s_real,
            num_points=args.num_points,
            normal_radius=args.normal_radius,
            normal_max_nn=args.normal_max_nn,
        )

        # 推理
        t0 = time.time()
        T_DL_norm = run_model_inference(model, t6, s6, device, args.num_reg_iter)
        total_time += time.time() - t0

        # 反归一化
        T_DL_real = denormalize_transform(T_DL_norm, centroid_t, scale)

        # DL-only 指标
        geo_rre, geo_rte = compute_registration_error(T_gt, T_DL_real)
        geo_errors.append((geo_rre, geo_rte))

        # DL + ICP
        if not args.no_icp:
            T_Final_real = refine_registration_icp(
                pts_s_real, pts_t_real, T_DL_real,
                distance_threshold=args.icp_threshold,
                max_iter=args.icp_max_iter,
            )
            icp_rre, icp_rte = compute_registration_error(T_gt, T_Final_real)
            icp_errors.append((icp_rre, icp_rte))

    # ── 汇总报告 ──
    g_r_mse, g_r_mae, g_t_mse, g_t_mae, g_succ = calculate_metrics(geo_errors)

    print(f"\nTotal inference time: {total_time:.2f}s  "
          f"({total_time / num_samples * 1000:.1f} ms/sample)")

    print("\n" + "★" * 108)
    print(f"🏆 {num_samples} 个样本评测报告 | angle={args.angle_range}°, trans={args.translation_range}")
    print(f"   成功标准: RTE < 0.001m 且 RRE < 1.0°")
    print("★" * 108)

    header = (
        f"{'评测阶段':<12} | "
        f"{'RRE MSE(°)':<12} {'RRE MAE(°)':<12} | {'RTE MSE(m)':<12} {'RTE MAE(m)':<12} | {'成功率(%)':<8}"
    )
    print(header)
    print("-" * 90)

    row_dl = (
        f"{'仅深度学习':<12} | "
        f"{g_r_mse:<12.4f} {g_r_mae:<12.4f} | {g_t_mse:<12.4f} {g_t_mae:<12.4f} | {g_succ:<8.2f}"
    )
    print(row_dl)

    if not args.no_icp and icp_errors:
        i_r_mse, i_r_mae, i_t_mse, i_t_mae, i_succ = calculate_metrics(icp_errors)
        row_icp = (
            f"{'DL + ICP':<12} | "
            f"{i_r_mse:<12.4f} {i_r_mae:<12.4f} | {i_t_mse:<12.4f} {i_t_mae:<12.4f} | {i_succ:<8.2f}"
        )
        print(row_icp)

    print("-" * 90)
    print("★" * 108 + "\n")


if __name__ == "__main__":
    main()