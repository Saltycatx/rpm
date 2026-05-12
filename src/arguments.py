"""Common arguments for train and evaluation for RPMNet"""
import argparse


def rpmnet_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', default='../logs', type=str,
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--name', type=str, help='Prefix to add to logging directory')
    parser.add_argument('--debug', action='store_true', help='If set, will enable autograd anomaly detection')
    # settings for input data_loader
    parser.add_argument('-i', '--dataset_path',
                        default='../datasets/modelnet40_ply_hdf5_2048',
                        type=str, metavar='PATH',
                        help='path to the processed dataset. Default: ../datasets/modelnet40_ply_hdf5_2048')
    parser.add_argument('--dataset_type', default='modelnet_hdf',
                        choices=['modelnet_hdf', 'surgical', 'bunny', 'armadillo', 'buddha', 'dragon'],
                        metavar='DATASET', help='dataset type (default: modelnet_hdf). '
                                                'Use \'surgical\' for the probe-trajectory P2C dataset.')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--noise_type', default='crop', choices=['clean', 'jitter', 'crop'],
                        help='Types of perturbation to consider')
    parser.add_argument('--rot_mag', default=45.0, type=float,
                        metavar='T', help='Maximum magnitude of rotation perturbation (in degrees)')
    parser.add_argument('--trans_mag', default=0.5, type=float,
                        metavar='T', help='Maximum magnitude of translation perturbation')
    parser.add_argument('--partial', default=[0.7, 0.7], nargs='+', type=float,
                        help='Approximate proportion of points to keep for partial overlap (Set to 1.0 to disable)')
    # Surgical dataset parameters (used when --dataset_type surgical) --------
    parser.add_argument('--template_points', type=int, default=2048,
                        help='(surgical) Number of template (reference) points to sample. Default: 2048')
    parser.add_argument('--source_points', type=int, default=2000,
                        help='(surgical) Number of probe-trajectory (source) points. Default: 1024')
    parser.add_argument('--angle_range', type=float, default=180.0,
                        help='(surgical) Max rotation angle in degrees for the rigid perturbation. Default: 180')
    parser.add_argument('--translation_range', type=float, default=2.0,
                        help='(surgical) Max translation magnitude per axis. Default: 2.0')
    parser.add_argument('--noise_sigma', type=float, default=0.005,
                        help='(surgical) Std-dev of Gaussian probe measurement noise. Default: 0.005')
    parser.add_argument('--coverage_ratio', type=float, default=0.90,
                        help='(surgical) Fraction of the half-surface covered by the probe path. Default: 0.90')
    # Model
    parser.add_argument('--method', type=str, default='rpmnet',
                        choices=['rpmnet', 'rpmnet_surgical'],
                        help='Model to use. \'rpmnet\' is the original model; '
                             '\'rpmnet_surgical\' enables all three P2C innovations '
                             '(T-PPF, P2C-Sinkhorn, SP-Loss).')
    # PointNet settings
    parser.add_argument('--radius', type=float, default=0.3, help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N', help='Max num of neighbors to use')
    # RPMNet settings
    parser.add_argument('--features', type=str, choices=['ppf', 'dxyz', 'xyz', 'tpf'],
                        default=['ppf', 'dxyz', 'xyz'],
                        nargs='+', help='Which features to use. Default: all standard features. '
                                        '\'tpf\' adds Tangent Point Pair Features (T-PPF) '
                                        'for trajectory source point clouds.')
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    parser.add_argument('--no_slack', action='store_true', help='If set, will not have a slack column.')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')
    parser.add_argument('--num_reg_iter', type=int, default=5,
                        help='Number of outer iterations used for registration (only during inference)')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae'], default='mae',
                        help='Loss to be optimized')
    parser.add_argument('--wt_inliers', type=float, default=1e-2, help='Weight to encourage inliers')
    # Surgical P2C innovations ---------------------------------------------------
    parser.add_argument('--src_slack_bias', type=float, default=-2.0,
                        help='(P2C-Sinkhorn) Log-space bias added to the source dust-bin column '
                             'before Sinkhorn iterations. Negative values (e.g. -2.0) make source '
                             'outliers less likely, enforcing the constraint that all probe-trajectory '
                             'points lie on the organ surface. Only active for rpmnet_surgical.'
                             ' Default: -2.0.')
    parser.add_argument('--wt_src_inliers', type=float, default=None,
                        help='(SP-Loss) Inlier penalty weight for source points. '
                             'Defaults to --wt_inliers. Set higher than --wt_ref_inliers '
                             'for partial-to-complete registration.')
    parser.add_argument('--wt_ref_inliers', type=float, default=None,
                        help='(SP-Loss) Inlier penalty weight for reference points. '
                             'Defaults to --wt_inliers. Set lower than --wt_src_inliers '
                             'for partial-to-complete registration (most organ surface is unvisited).')
    # Training parameters
    parser.add_argument('--train_batch_size', default=4, type=int, metavar='N',
                        help='training mini-batch size (default 8)')
    parser.add_argument('-b', '--val_batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size during validation or testing (default: 16)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--gpu', default=0, type=int, metavar='DEVICE',
                        help='GPU to use, ignored if no GPU is present. Set to negative to use cpu')
    return parser


def rpmnet_train_arguments():
    """Used only for training"""
    parser = argparse.ArgumentParser(parents=[rpmnet_arguments()])

    parser.add_argument('--train_categoryfile', type=str, metavar='PATH', default='./data_loader/modelnet40_half1.txt',
                        help='path to the categories to be trained')  # eg. './dataset/modelnet40_half1.txt'
    parser.add_argument('--val_categoryfile', type=str, metavar='PATH', default='./data_loader/modelnet40_half1.txt',
                        help='path to the categories to be val')  # eg. './sampledata/modelnet40_half1.txt'
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate during training')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--summary_every', default=200, type=int, metavar='N',
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('--validate_every', default=-4, type=int, metavar='N',
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative).'
                             'Also saves checkpoints at the same interval')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data_loader loader (default: 4).')
    parser.add_argument('--num_train_reg_iter', type=int, default=2,
                        help='Number of outer iterations used for registration (only during training)')

    parser.description = 'Train RPMNet'
    return parser


def rpmnet_eval_arguments():
    """Used during evaluation"""
    parser = argparse.ArgumentParser(parents=[rpmnet_arguments()])

    # settings for input data_loader
    parser.add_argument('--test_category_file', type=str, metavar='PATH', default='./data_loader/modelnet40_half2.txt',
                        help='path to the categories to be val')
    # Provided transforms
    parser.add_argument('--transform_file', type=str,
                        help='If provided, will use transforms from this provided pickle file')
    # Save out evaluation data_loader for further analysis
    parser.add_argument('--eval_save_path', type=str, default='../eval_results',
                        help='Output data_loader to save evaluation results')

    parser.description = 'RPMNet evaluation'
    return parser
