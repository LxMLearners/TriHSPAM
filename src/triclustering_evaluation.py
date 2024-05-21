import numpy as np
import math

from scipy.optimize import linear_sum_assignment

# =================== CE Auxiliary Functions =================================

# CE aux


def get_clusters_points(triclusters):

    points = dict()
    for k in triclusters.keys():
        rows = triclusters[k][0]
        cols = triclusters[k][1]
        ctxs = triclusters[k][2]
        points[k] = [(row, col, ctx)
                     for row in rows for col in cols for ctx in ctxs]

    return points

# CE aux


def calculate_confusion_matrix(solution1, solution2):

    confusion_matrix = np.zeros((len(solution1), len(solution2)))

    for s1 in range(0, len(solution1)):
        for s2 in range(0, len(solution2)):
            s1_key = list(solution1.keys())[s1]
            s2_key = list(solution2.keys())[s2]
            confusion_matrix[s1][s2] = len(set(solution1[s1_key])
                                           .intersection(solution2[s2_key]))

    return confusion_matrix

# CE aux


def support_matrix(solution, nrows, ncols, nctxs):

    matrix = np.zeros((nrows, ncols, nctxs))

    for k in solution.keys():
        for row, col, ctx in solution[k]:
            matrix[row][col][ctx] += 1

    return matrix

# CE aux


def calculate_union_size(solution1, solution2, nrows, ncols, nctxs):

    s1 = support_matrix(solution1, nrows, ncols, nctxs)
    s2 = support_matrix(solution2, nrows, ncols, nctxs)

    return np.maximum(s1, s2).sum()

# =================== Recoverability Auxiliary Functions ====================


def intersection_rate(t, s):

    # print('S:', s)

    gt_rows = set(t[0])
    gt_cols = set(t[1])
    gt_ctxs = set(t[2])

    rows_intersect = gt_rows.intersection(set(s[0]))
    cols_intersect = gt_cols.intersection(set(s[1]))
    ctxs_intersect = gt_ctxs.intersection(set(s[2]))

    shared_elems = len(rows_intersect) * \
        len(cols_intersect) * len(ctxs_intersect)
    dim = len(gt_rows) * len(gt_cols) * len(gt_ctxs)

    return shared_elems / dim


def calculate_mu(data):

    d_IJK = data.mean()
    d_iJK = np.full(data.shape[1], np.inf)
    d_IjK = np.full(data.shape[2], np.inf)
    d_IJk = np.full(data.shape[0], np.inf)
    mu_ijk = 0

    for k in range(0, data.shape[0]):
        for i in range(0, data.shape[1]):
            for j in range(0, data.shape[2]):

                if d_iJK[i] == np.inf:
                    d_iJK[i] = data[:, [i], :].mean()

                if d_IjK[j] == np.inf:
                    d_IjK[j] = data[:, :, [j]].mean()

                if d_IJk[k] == np.inf:
                    d_IJk[k] = data[k].mean()

                mu_ijk += pow((data[k][i][j] - d_iJK[i] -
                              d_IjK[j] - d_IJk[k] + 2 * d_IJK), 2)

    return mu_ijk


def MSR3D(data):
    """
    data: np.array 3D with the tricluster values
    """

    data_dim = data.shape[0] * data.shape[1] * data.shape[2]
    mu_ijk = calculate_mu(data)
    return (1 / data_dim) * mu_ijk


def calc_observations_pc(data, t):

    mr_t = data[t].mean(axis=0)
    pc = np.nan_to_num([np.corrcoef(data[t, r, :], mr_t)[0, 1]
                       for r in range(0, data.shape[1])], nan=1)
    return pc.mean()


def IntraTemporalHomogeneity(data):
    """
    data: np.array 3D with the tricluster values
    """
    avg_pc = np.array([calc_observations_pc(data, t)
                      for t in range(0, data.shape[0])])
    return avg_pc.mean()


def InterTemporalHomogeneity(data):
    """
    data: np.array 3D with the tricluster values
    """
    mmr = data.mean(axis=0).mean(axis=0)
    pc = np.nan_to_num([np.corrcoef(data[t].mean(axis=0), mmr)[0, 1] for t in range(0, data.shape[0])],
                       nan=1)

    return np.array(pc).mean()


def Recoverability(ground_truth, solution, verbose=False):
    """
    ground_truth: dict in format {id:[[rows],[cols],[ctxs]]}
    solution: dict in format {id:[[rows],[cols],[ctxs]]}
    """
    res = dict()

    for i, t in ground_truth.items():
        if verbose:
            rate = []
            print("Solution:", t)
            for s in solution.values():
                i_r = intersection_rate(t, s)
                print(f"Tr:{s}, Rec:{round(i_r, 2)}")
                rate.append(i_r)
            print(round(max(rate), 2))
            res[i] = max(rate)
        else:
            res[i] = 0 if len(solution) == 0 else round(
                max([intersection_rate(t, s) for s in solution.values()]), 2)
    return res

# Clustering Error 3D


def ClusteringError3D(ground_truth, clust_solution, nrows, ncols, nctxs):
    """
    ground_truth: dict in format {id:[[rows],[cols],[ctxs]]}
    clust_solution: dict in format {id:[[rows],[cols],[ctxs]]}
    nrows: int with nr of rows
    ncols: int with nr of cols
    nctxs: int with nr of ctxs
    """
    ground_truth = get_clusters_points(ground_truth)
    clust_solution = get_clusters_points(clust_solution)

    confusion_matrix = calculate_confusion_matrix(ground_truth, clust_solution)
    confusion_matrix = np.negative(confusion_matrix)

    row_ind, col_ind = linear_sum_assignment(confusion_matrix)
    d_max = -confusion_matrix[row_ind, col_ind].sum()

    union_size = calculate_union_size(
        ground_truth, clust_solution, nrows, ncols, nctxs)

    error = (union_size - d_max) / union_size

    return error


def overlaping_size(t, s):
    gt_rows = set(t[0])
    gt_cols = set(t[1])
    gt_ctxs = set(t[2])

    rows_intersect = gt_rows.intersection(set(s[0]))
    cols_intersect = gt_cols.intersection(set(s[1]))
    ctxs_intersect = gt_ctxs.intersection(set(s[2]))

    shared_elems = len(rows_intersect) * \
        len(cols_intersect) * len(ctxs_intersect)
    return shared_elems


def jaccard_score(T1, T2):
    size_T1 = len(T1[0]) * len(T1[1]) * len(T1[2])
    size_T2 = len(T2[0]) * len(T2[1]) * len(T2[2])
    overlap = overlaping_size(T1, T2)

    jac_score = (size_T1 + size_T2 - (size_T1 + size_T2 - overlap)
                 ) / (size_T1 + size_T2 - overlap)
    return jac_score


def RMS3(ground_truth, clust_solution):
    max_jac_trics = list()
    for t_s in clust_solution.values():
        max_score = 0
        max_t_id = 0
        for t_id, t_gt in ground_truth.items():
            score = jaccard_score(t_s, t_gt)
            if score >= max_score:
                max_score = score
                max_t_id = t_id
        max_jac_trics.append(max_t_id)

    # rms3d = 0
    rms3 = dict()
    clust_solution_list = list(clust_solution.values())
    for t_s_i in range(len(clust_solution_list)):
        max_t_gt = ground_truth[max_jac_trics[t_s_i]]

        t_I = set(clust_solution_list[t_s_i][0])
        t_J = set(clust_solution_list[t_s_i][1])
        t_K = set(clust_solution_list[t_s_i][2])

        t_gt_I = set(max_t_gt[0])
        t_gt_J = set(max_t_gt[1])
        t_gt_K = set(max_t_gt[2])

        shared_I = len(t_I.intersection(t_gt_I))
        shared_J = len(t_J.intersection(t_gt_J))
        shared_K = len(t_K.intersection(t_gt_K))

        union_I = len(t_I.union(t_gt_I))
        union_J = len(t_J.union(t_gt_J))
        union_K = len(t_K.union(t_gt_K))
        
        rms3[t_s_i] = ((shared_I * shared_J * shared_K) /
                  (union_I * union_J * union_K)) ** (1./3)
        # rms3d += ((shared_I * shared_J * shared_K) /
        #           (union_I * union_J * union_K)) ** (1./3)

    return rms3




## Heterogeneous triclusters eval
def h_var3(data, numeric_cols_idc, symbolic_cols_idc):
    """
    data: np.array 3D with the tricluster values
    """
    numeric_component = data[:, :, numeric_cols_idc]
    symbolic_component = data[:, :, symbolic_cols_idc]

    numeric_component = numeric_component.astype(float)

    numeric_metric = coeficient_variance_numeric(numeric_component) / numeric_component.size if numeric_component.size > 0 else 0
    symbolic_metric = gini_impurity_3D(symbolic_component) / symbolic_component.size if symbolic_component.size > 0 else 0
    return numeric_metric + symbolic_metric + missing_values_exp(data)


def variance_numeric(data):

    variance_x = np.var(data, axis=1)

    # Calculate the average variance for each slice along the X axis
    avg_variances_x = np.mean(variance_x, axis=0)

    # Calculate the overall average of the average variances
    avg_of_avg_variances = np.mean(avg_variances_x)

    return avg_of_avg_variances

def coeficient_variance_numeric(array_3d):
    """
    Computes the average coefficient of variation of each slice along the X axis of a 3D NumPy array.

    Parameters:
    - array_3d: NumPy array with shape (n, m, p)

    Returns:
    - avg_of_avg_coeff_of_variation: Average of average coefficients of variation along the X axis
    """
    if array_3d.size == 0:
        return 0
    
    std_deviation_x = np.nanstd(array_3d, axis=1)

    mean_x = np.nanmean(array_3d, axis=1)

    # Calculate the coefficient of variation for each slice along the X axis
    coefficient_of_variation_x = std_deviation_x / mean_x

    # Calculate the average coefficient of variation along the X axis
    avg_coeff_of_variation_x = np.mean(coefficient_of_variation_x, axis=0)

    # Calculate the overall average of the average coefficients of variation
    avg_of_avg_coeff_of_variation = np.mean(avg_coeff_of_variation_x)

    return avg_of_avg_coeff_of_variation


def gini_impurity_3D(data):
    """
    Computes the average Gini impurity of each slice along the X axis of a 3D NumPy array.

    Parameters:
    - array_3d: NumPy array with shape (n, m, p), containing characters

    Returns:
    - avg_of_avg_gini_impurity: Average of average Gini impurities along the X axis
    """

    # Function to calculate Gini impurity for a given array of labels
    def calculate_gini(labels):
        labels_mask = [True if isinstance(x, str) else not math.isnan(x) for x in labels]
        labels = labels[labels_mask]

        class_counts = np.unique(labels, return_counts=True)[1]
        total_instances = len(labels)
        gini = 1.0 - np.sum((class_counts / total_instances) ** 2)
        return gini
    
    if data.size == 0:
        return 0
    # Calculate Gini impurity along the X axis
    gini_x = np.apply_along_axis(lambda x: calculate_gini(x), axis=1, arr=data)

    # Calculate the average Gini impurity for each slice along the X axis
    avg_gini_x = np.mean(gini_x, axis=0)

    # Calculate the overall average of the average Gini impurities
    avg_of_avg_gini_impurity = np.mean(avg_gini_x)

    return avg_of_avg_gini_impurity


def missing_values_exp(data):

    flattened_array = data.flatten()
    missing_values_count = np.sum([1 for elem in flattened_array if not isinstance(elem, str) and math.isnan(elem)])
    
    # missing_values_count = np.count_nonzero(np.isnan(flattened_array.astype(float)) | 
    #                                         (flattened_array == None) | 
    #                                         (flattened_array == ''))

    return missing_values_count / data.size
