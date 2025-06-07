import numpy as np


def current_sol_1(dist, P1, P2):
    ### Error handling
    assert dist >= 0, "Distance cannot be negative"
    assert (align_1.shape[0] == 3 or align_1.shape[1] == 3) and len(align_1.shape) == 2, "Shape must be (3, N) or (N, 3)"
    assert align_1.shape == align_2.shape, "Arrays must have the same shape"

    ### Makes sure shape is correct
    if align_1.shape[1] == 3:
            align_1 = align_1.T
            align_2 = align_2.T

    ### Number of points
    n = P1.shape[1]
    
    ### Fills NxN matrix with ones
    check_mat = np.ones((n, n))

    ### Removes ones in the diagonal
    np.fill_diagonal(check_mat, 0)

    ### Finds the indices of all points
    i, j = np.where(check_mat == 1)

    ### Calculates vector for the calculation of minimum distance between two points during their interpolation
    p0_ij = P1[:, j] - P1[:, i] 
    p1_ij = P2[:, j] - P2[:, i]
    
    ### Takes the norm
    a = np.linalg.norm(p0_ij, axis=0) 
    b = np.linalg.norm(p1_ij, axis=0)

    ### Finds the dot product
    dot_prod = dot_prod = np.sum(p0_ij * p1_ij, axis=0)
    denom = (a*a + b*b - 2*dot_prod)    
    
    t_min = (a*a - dot_prod) / (denom)

    ### Clips t to appropriate interval
    t_min = np.clip(t_min, 0, 1)

    ### Finds the minimum distance
    m = np.sqrt((1 - t_min)**2*a**2 + t_min**2*b**2 + 2*t_min*(1-t_min)*dot_prod)
    
    ### Finds the indices of the minimum distances below the maximum allowed distance
    m_idx = np.where(m <= dist)

    ### Finds the index of the points clashing
    i_int = i[m_idx]
    j_int = j[m_idx]
        

    ### Makes a list of points steric clashes 
    steric_cl = list(zip(i_int, j_int))
    steric_cl = np.unique(np.array([sorted(i) for i in steric_cl]), axis=0)        

    return steric_cl
    
