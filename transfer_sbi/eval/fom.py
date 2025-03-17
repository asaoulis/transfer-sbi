import numpy as np

def compute_fom(samples):
    """
    Compute the Figure of Merit (FoM) given posterior samples.

    Parameters:
    samples (numpy.ndarray): An (N, d) array where N is the number of samples
                             and d is the number of parameters.

    Returns:
    float: The computed Figure of Merit (FoM).
    """
    cov_matrix = np.cov(samples, rowvar=False)  # Compute covariance matrix
    det_cov = np.linalg.det(cov_matrix)  # Determinant of covariance matrix
    
    if det_cov <= 0:
        raise ValueError("Covariance matrix determinant must be positive for FoM computation.")
    
    fom = 1.0 / np.sqrt(det_cov)  # Compute Figure of Merit
    return fom