import numpy as np

from .processing_tools_final import normalize_feature_sequence_z
from .feature_extraction_tools import ExtractFeatureMatrix

#### Code from LIBFMP ####
#   https://github.com/meinardmueller/libfmp/blob/master/libfmp/c4/c4s2_ssm.py
#   Error when loading the app due to a dependency from libfmp - soundfile???
# @jit(nopython=True)
def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False): # kernel=None,for incase using a different karnel
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = int(S.shape[0])
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    #S_padded = np.pad(S, L, mode='constant')
    L1 = int(L)

    # np.pad expects a tuple for pad_width if S is 2D. We pad both dimensions equally.
    S_padded = np.pad(S, L1, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[int(n)] = np.sum(S_padded[int(n):int(n) + M, int(n):int(n) + M] * kernel)

    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

#@jit(nopython=True)
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S

def compute_kernel_checkerboard_gaussian(L, var=1.0, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel
####

def compute_ssm(s, window_size, overlap_perc):                          # s = channels
    feat_Mat = ExtractFeatureMatrix(s, window_size, perc_overlap=overlap_perc)

    F_set = normalize_feature_sequence_z(np.array(feat_Mat["allfeatures"]))     # Do we need to normalize the features?

    S = compute_sm_dot(F_set, F_set)

    return S

def compute_novelty(S, kernel_size):
    nov_ssm = compute_novelty_ssm(S, L = kernel_size)

    return nov_ssm

def novelty_event_cost(s, window_size, kernel_size, overlap_perc):
    S = compute_ssm(s, window_size, overlap_perc)
    nov_ssm = compute_novelty(S, kernel_size)

    return nov_ssm

def perioric_event_cost(s, window_size, overlap_perc):
    S = compute_ssm(s, window_size, overlap_perc)
    per_ssm = np.sum(S, axis=0)

    return per_ssm