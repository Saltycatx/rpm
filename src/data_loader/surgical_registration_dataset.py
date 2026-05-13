import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


class SurgicalRegistrationData(Dataset):
    """Surgical probe-trajectory to complete-surface registration dataset.

    Produces pairs of

    * **source** - a continuous, arc-length-resampled probe trajectory that
      covers a random half of the object surface, perturbed with Gaussian
      measurement noise and then rigidly transformed by a random SE(3).
    * **reference** - the full, untransformed object surface (template).

    ``__getitem__`` returns a dict that is directly compatible with the
    RPMNet training and evaluation pipeline:

    * ``'points_src'``   - (N_source, 6)  float32  XYZ + surface normals of the
      transformed probe trajectory.
    * ``'points_ref'``   - (N_template, 6) float32  XYZ + surface normals of the
      complete organ template.
    * ``'transform_gt'`` - (3, 4)          float32  Ground-truth rigid transform
      that maps *source to reference* frame (i.e. the inverse of the applied
      forward transform).
    * ``'points_raw'``   - (N_template, 6) float32  Clean copy of the template,
      used by the Chamfer-distance metric.
    * ``'label'``        - ()              int64    Object category index.
    """

    def __init__(
        self,
        data_root="./learning3d/data/modelnet40_ply_hdf5_2048",
        partition='test',
        template_points=2048,
        source_points=1024,
        angle_range=180,
        translation_range=2.0,
        noise_sigma=0.005,
        coverage_ratio=0.90,
        normal_k=16,
    ):
        super().__init__()
        self.template_points = template_points
        self.source_points = source_points
        self.angle_range_rad = angle_range * (np.pi / 180)
        self.translation_range = translation_range
        self.noise_sigma = noise_sigma
        self.coverage_ratio = coverage_ratio
        self.normal_k = normal_k

        self.data, self.labels = self._load_data(data_root, partition)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, data_root, partition):
        """Load XYZ point cloud data from ModelNet40 HDF5 files."""
        data, labels = [], []
        h5_files = sorted(glob.glob(os.path.join(data_root, f'ply_data_{partition}*.h5')))
        if not h5_files:
            raise FileNotFoundError(
                f"No {partition} HDF5 files found in {data_root}."
            )

        for h5_name in h5_files:
            with h5py.File(h5_name, 'r') as f:
                data.append(f['data'][:].astype('float32'))
                labels.append(f['label'][:].astype('int64'))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0).flatten()
        return data, labels

    def __len__(self):
        return self.data.shape[0]

    # ------------------------------------------------------------------
    # Normal estimation via local PCA
    # ------------------------------------------------------------------

    def _estimate_normals_pca(self, points):
        """Estimate surface normals using PCA on k-nearest neighbours.

        For each point the smallest eigenvector of the local covariance matrix
        (built from ``normal_k`` neighbours) is taken as the normal direction.
        This is a standard, rotation-equivariant estimator: rotating the point
        cloud rotates the normals consistently.

        Args:
            points: (N, 3) float32 numpy array.

        Returns:
            normals: (N, 3) float32 numpy array of unit normals.
        """
        N = points.shape[0]
        normals = np.zeros((N, 3), dtype=np.float32)
        k = min(self.normal_k, N - 1)
        tree = KDTree(points)
        _, indices = tree.query(points, k=k + 1)  # +1 because query includes self

        for i in range(N):
            nbrs = points[indices[i, 1:]]  # exclude the query point itself
            if nbrs.shape[0] < 3:
                normals[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                continue
            cov = np.cov(nbrs.T)
            if cov.ndim < 2:
                normals[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                continue
            _, eigvecs = np.linalg.eigh(cov)
            n = eigvecs[:, 0]  # eigenvector for the smallest eigenvalue
            normals[i] = (n / (np.linalg.norm(n) + 1e-8)).astype(np.float32)

        return normals

    # ------------------------------------------------------------------
    # Transform generation
    # ------------------------------------------------------------------

    def _generate_transform(self):
        """Sample a random SE(3) rigid transform.

        Returns:
            R           - (3, 3) float32 rotation matrix
            translation - (3,)   float32 translation vector
            igt         - (4, 4) float32 forward transform (source_raw -> source_t)
        """
        anglex = np.random.uniform(-1, 1) * self.angle_range_rad
        angley = np.random.uniform(-1, 1) * self.angle_range_rad
        anglez = np.random.uniform(-1, 1) * self.angle_range_rad
        translation = np.array([
            np.random.uniform(-self.translation_range, self.translation_range),
            np.random.uniform(-self.translation_range, self.translation_range),
            np.random.uniform(-self.translation_range, self.translation_range),
        ], dtype=np.float32)

        rotation = Rotation.from_euler('zyx', [anglez, angley, anglex])
        R = rotation.as_matrix().astype(np.float32)
        igt = np.eye(4, dtype=np.float32)
        igt[:3, :3] = R
        igt[:3, 3] = translation

        return R, translation, igt

    # ------------------------------------------------------------------
    # Probe-path generation helpers
    # ------------------------------------------------------------------

    def _random_half_cut(self, points):
        """Keep roughly half the surface points using a random cutting plane."""
        centroid = np.mean(points, axis=0)
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)

        dots = np.dot(points - centroid, normal)
        keep_mask = dots > 0
        half_points = points[keep_mask]

        if len(half_points) < len(points) * 0.2:
            indices = np.argsort(dots)
            half_points = points[indices[len(points) // 2:]]

        return half_points

    def _generate_continuous_probe_path(self, points, coverage_ratio, num_source_pts):
        """Generate a single non-branching probe trajectory on the surface.

        Uses a momentum-guided greedy walk followed by arc-length resampling to
        produce a non-homologous (interpolated) point cloud that simulates
        high-frequency probe sampling.
        """
        N = len(points)
        target_len = int(N * coverage_ratio)

        visited = np.zeros(N, dtype=bool)
        start_idx = np.argmax(points[:, 0])
        visited[start_idx] = True
        path = [start_idx]

        curr_idx = start_idx
        current_dir = np.array([1.0, 0.0, 0.0])
        unvisited_idx = np.where(~visited)[0]

        for _ in range(target_len - 1):
            curr_pt = points[curr_idx]
            vecs = points[unvisited_idx] - curr_pt
            dists = np.linalg.norm(vecs, axis=1)

            k = min(15, len(unvisited_idx))
            if k == 0:
                break

            idx_k = np.argpartition(dists, k - 1)[:k]
            dists_k = dists[idx_k]
            vecs_k = vecs[idx_k]

            norms_k = dists_k + 1e-8
            vecs_norm_k = vecs_k / norms_k[:, None]
            cos_theta_k = np.dot(vecs_norm_k, current_dir)

            d_min, d_max = dists_k.min(), dists_k.max()
            if d_max > d_min:
                dists_k_norm = (dists_k - d_min) / (d_max - d_min)
            else:
                dists_k_norm = np.zeros_like(dists_k)

            scores = dists_k_norm - 0.8 * cos_theta_k
            best_k = np.argmin(scores)
            chosen_unvisited_idx = idx_k[best_k]
            next_idx = unvisited_idx[chosen_unvisited_idx]

            step_vec = points[next_idx] - curr_pt
            step_dir = step_vec / (np.linalg.norm(step_vec) + 1e-8)
            current_dir = 0.5 * current_dir + 0.5 * step_dir
            current_dir /= np.linalg.norm(current_dir)

            curr_idx = next_idx
            visited[curr_idx] = True
            path.append(curr_idx)

            unvisited_idx[chosen_unvisited_idx] = unvisited_idx[-1]
            unvisited_idx = unvisited_idx[:-1]

        path_points = points[path]

        # Arc-length resampling: produces non-homologous coords along the path
        if len(path_points) >= 2:
            diffs = np.diff(path_points, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = cum_length[-1]

            if total_length > 1e-8:
                target_lengths = np.linspace(0, total_length, num_source_pts)
                source_pts = np.zeros((num_source_pts, 3), dtype=np.float32)
                for dim in range(3):
                    source_pts[:, dim] = np.interp(
                        target_lengths, cum_length, path_points[:, dim])
                return source_pts

        idx = np.random.choice(len(path_points), num_source_pts, replace=True)
        return path_points[idx].copy()

    # ------------------------------------------------------------------
    # Dataset item
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        """Return one RPMNet-compatible training sample as a dict of numpy arrays.

        The returned dict contains:

        * ``'points_src'``   - transformed probe trajectory XYZ + normals  (N_source, 6)
        * ``'points_ref'``   - complete template XYZ + normals              (N_template, 6)
        * ``'transform_gt'`` - ground-truth *source -> reference* transform (3, 4)
        * ``'points_raw'``   - clean copy of the template                   (N_template, 6)
        * ``'label'``        - category index                               scalar int64

        Transform convention (consistent with RPMNet se3.transform):
            se3.transform(transform_gt, points_src[:, :3]) ~= source_raw
        where source_raw are the probe points before the rigid perturbation.
        """
        full_xyz = self.data[index][:, :3]  # XYZ only from stored data

        # ---- Template (reference): complete object surface ----
        if len(full_xyz) > self.template_points:
            t_idx = np.random.choice(len(full_xyz), self.template_points, replace=False)
            template_xyz = full_xyz[t_idx].copy()
        else:
            template_xyz = full_xyz[:self.template_points].copy()

        # ---- Probe path generation ----
        half_template = self._random_half_cut(template_xyz)
        source_raw = self._generate_continuous_probe_path(
            half_template, self.coverage_ratio, self.source_points)

        # ---- Measurement noise ----
        if self.noise_sigma > 0:
            source_raw += np.random.normal(
                0, self.noise_sigma, source_raw.shape).astype(np.float32)

        # ---- Rigid transform: source_t = R @ source_raw + translation ----
        R, translation, _ = self._generate_transform()
        source_t = (R @ source_raw.T).T + translation  # (N_source, 3)

        # ---- Normal estimation ----
        # For the template (surface): PCA gives reliable surface normals.
        # For the trajectory: source points lie ON the surface (sampled from it),
        # so local PCA on the transformed neighbours still yields valid surface
        # normals (in the transformed frame), consistent with PPF computation.
        template_normals = self._estimate_normals_pca(template_xyz)  # (N_template, 3)
        source_normals = self._estimate_normals_pca(source_t)         # (N_source, 3)

        # ---- Pack 6-channel (xyz + normals) arrays ----
        points_ref = np.concatenate(
            [template_xyz, template_normals], axis=-1).astype(np.float32)
        points_src = np.concatenate(
            [source_t, source_normals], axis=-1).astype(np.float32)

        # ---- Ground-truth transform: inverse of the forward transform ----
        # RPMNet se3.transform(g, a) = a @ R^T + t
        # We need: source_t @ R_inv^T + t_inv = source_raw
        #   => R_inv = R^T,  t_inv = -R^T @ translation
        R_inv = R.T.astype(np.float32)
        t_inv = (-R.T @ translation).astype(np.float32)
        transform_gt = np.concatenate([R_inv, t_inv[:, None]], axis=1)  # (3, 4)

        return {
            'points_src': points_src,           # (N_source,   6) float32
            'points_ref': points_ref,           # (N_template, 6) float32
            'transform_gt': transform_gt,       # (3, 4)          float32
            'points_raw': points_ref.copy(),    # (N_template, 6) float32 - clean template
            'label': np.array(self.labels[index], dtype=np.int64),
        }


# ===================== Visualization check =====================
if __name__ == '__main__':
    dataset = SurgicalRegistrationData(
        data_root="./learning3d/data/modelnet40_ply_hdf5_2048",
        partition='test',
        template_points=2048,
        source_points=2048,
        angle_range=90,
        translation_range=1.0,
        noise_sigma=0.000,
        coverage_ratio=0.90,
    )

    print(f"Dataset size: {len(dataset)}")

    import open3d as o3d

    for i in range(3):
        sample = dataset[i]
        t_np = sample['points_ref'][:, :3]
        s_np = sample['points_src'][:, :3]
        transform_gt = sample['transform_gt']   # (3, 4): [R_inv | t_inv]

        # Recover source in reference frame: se3.transform(transform_gt, source_t)
        #   = source_t @ R_inv^T + t_inv
        R_inv = transform_gt[:, :3]   # (3, 3)
        t_inv = transform_gt[:, 3]    # (3,)
        source_unwarped = s_np @ R_inv.T + t_inv

        t_pcd = o3d.geometry.PointCloud()
        t_pcd.points = o3d.utility.Vector3dVector(t_np)
        t_pcd.paint_uniform_color([0.8, 0.8, 0.8])

        s_pcd = o3d.geometry.PointCloud()
        s_pcd.points = o3d.utility.Vector3dVector(source_unwarped)
        s_pcd.paint_uniform_color([0, 0, 1])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0

        print(f"Sample {i}: continuous non-homologous probe path.")
        o3d.visualization.draw_geometries(
            [t_pcd, s_pcd],
            window_name=f"Sample {i} - Continuous Non-homologous Path"
        )
