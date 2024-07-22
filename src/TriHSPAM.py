import time
import subprocess
import os
import numpy as np
import trihspam_utils as utils
from pathlib import Path
from triclustering_evaluation import h_var3
from sklearn.preprocessing import MinMaxScaler

class TriHSPAM:
    """ TriHSPAM Algorithm (Soares et al., 2024)

        Triclustering algorithm to mine triclusters from temporal
        and heterogeneous 3W data.
    
        Parameters
        ----------

        symb_features_idx : list
            The symbolic columns (features) indices w.r.t the input data
        
        num_features_idx : list
            The numeric columns (features) w.r.t the input data  
        
        min_I : int
            The triclusters' minimum number of observations (rows) 

        min_J : int
            The triclusters' minimum number of features (columns) 
        
        min_K : int
            The triclusters' minimum number of contexts 
        
        time_as_cols : bool, default=True
            It defines the orientation of the input data. It should be True if time points 
            are represented in the X axis of the input data.
        
        disc_method : {'eq_size', 'eq_width'}, default=`'eq_size'`
            Selects the discretization method to be applied in numeric features.
            May be 'eq_size' or 'eq_width.

        n_bins : int, default=5
            The number of bins to consider in discretization

        mv_method : {'locf', None}, default=None
            The missing values handling method.
            Options available:
                - `locf` for last observation carried forward
                - `None` to ignore missing values
        
        spm_algo : {'fournier08closed', 'prefixspan','clospan', 'spam'}, default='fournier08closed'
            The sequential pattern mining algorithm to be used to mine the triclusters.
        
        coerence_threshold : float
            The admissible threshold for merit function (h3_var)

        Attributes
        ----------
        
        
    """

    def __init__(
            self, 
            symb_features_idx, 
            num_features_idx, 
            min_I, 
            min_J, 
            min_K,
            time_as_cols = True, 
            disc_method='eq_size', 
            n_bins=5,
            mv_method=None, 
            spm_algo="fournier08closed",
            time_relaxed=False, 
            coerence_threshold=0.5
        ):
        
        self._available_algo = {
            'clospan' : 'CloSpan',
            'spam'    :'SPAM_AGP',
            'prefixspan'    : 'PrefixSpan_AGP',
            'fournier08closed': 'Fournier08-Closed+time'
        }
        self._symb_features_idx = symb_features_idx
        self._num_features_idx = num_features_idx
        self._min_I = min_I
        self._min_J = min_J
        self._min_K = min_K
        self._time_as_cols = time_as_cols
        self._disc_method = disc_method
        self._n_bins = n_bins
        self._mv_method = mv_method
        self._spm_algo = spm_algo
        self._relaxed = time_relaxed
        self._coer_thresh = coerence_threshold

    def set_discretization(self, discretization):
        """
        Indicates a personalized discretization for specific numeric features.
        
        Parameters:
        - discretization: a dict with the discretized values allowed for each feature.
            Its keys are the indices of features in the input data to be fitted.
            Its values are dicts whose keys are the discrete symbol and values are list with 
                the corresponding numbers
        """
        self._abstractions = discretization

    def _validate_arguments(self):
        pass

    def fit(self, data):
        """Create a triclustering for data.

        Parameters
        ----------
        X : array-like of shape (n_features, n_observations, n_contexts) if time_as_cols, 
                        or shape (n_contexts, n_observations, n_features) otherwise
            Training data.

        Returns
        -------
        self : object
            TriHSPAM instance.
        """
        if not self._time_as_cols:
            self._data = np.transpose(data, (2,1,0))
        else:
            self._data = data

        #Handling Missing Values
        if self._mv_method == "locf":
            self._data = utils.impute_missing_with_locf(self._data)
        
        num_cube = self._data[self._num_features_idx , :, :]
        num_cube = num_cube.astype(float)
        symb_cube = self._data[self._symb_features_idx, :, :]
        # print(num_cube)
        # scaler = MinMaxScaler()
        # num_cube = scaler.fit_transform(num_cube.reshape(-1, num_cube.shape[-1])).reshape(num_cube.shape)
        # print(num_cube)

        if not hasattr(self, "_abstractions"):
            #Discretization
            self._abstractions = {}

        if self._disc_method == "eq_size":
            num_abst = utils.equal_freq_binning_np(num_cube, self._n_bins)
        elif self._disc_method == "eq_width":
            num_abst = utils.equal_width_binning_np(num_cube, self._n_bins)

        symb_cube = symb_cube.astype(object)
        symb_abst = utils.unique_values_abstractions(symb_cube)

        f_i = 0
        for _ in self._data:
            if f_i not in self._abstractions:
                if f_i in self._num_features_idx:
                    self._abstractions[f_i] = num_abst
                else:
                    self._abstractions[f_i] = symb_abst
            f_i += 1

        f_labels = [f"f{i}" for i in range(len(self._data))]
        types = {}
        types['numeric'] = [f_labels[i] for i in self._num_features_idx]
        types['symbolic'] = [f_labels[i] for i in self._symb_features_idx]

        self._mss = utils.data_to_instant_mss(self._data, f_labels, types, self._abstractions, self._relaxed)

        K = len(self._data[0][0])
        self._mss_subjects = {}
        for k, v in self._mss.items():
            seq = utils.restrictive_coocurences(v, K)
            self._mss_subjects[k] = seq
    
        Path("temps/").mkdir(parents=True, exist_ok=True)
        input_temp_file = time.strftime("temps/spm_%Y%m%d-%H%M%S.txt")
        output_temp_file = time.strftime("temps/out_%Y%m%d-%H%M%S.txt")

        utils.seqs_to_spmf(self._mss_subjects, 
                           input_temp_file, time_constr=self._spm_algo=='fournier08closed')

        percent_min_I = int((self._min_I/self._data.shape[1])*100)
        
        if self._spm_algo != "fournier08closed":
            subprocess.run(
                f'java -jar spmf_vd.jar run {self._available_algo[self._spm_algo]} '
                + f'{input_temp_file} {output_temp_file} {percent_min_I}% true', 
                shell=True)
        else:
            subprocess.run(
                f'java -jar spmf_vd.jar run {self._available_algo[self._spm_algo]} '
                    + f'{input_temp_file} {output_temp_file} {percent_min_I}% 1 1 '
                    + f'{self._min_K-1} {self._data.shape[2]-1}', 
                shell=True)

        os.remove(input_temp_file)

        self._patterns = utils.get_subjects_patterns(output_temp_file, f_labels, self._abstractions)

        os.remove(output_temp_file)

        self._triclusters_type = dict()
        for pattern, observations in self._patterns.items():
            pattern = pattern.strip(" ")
            rows_I = list(map(int, observations))
            cols_J = list()
            contx_K = list()
            cols_contx_JK = utils.extract_features(pattern)
            if self._relaxed:
                contx_K = dict()
                for jk in cols_contx_JK:
                    j = int(jk.replace("f",""))
                    if j not in cols_J:
                        cols_J.append(j)
                        cols_J.sort()
                for ri in rows_I:
                    contx_K[ri] = utils.find_tp_pattern(self._mss_subjects[f"X{ri}"], pattern)
                candidate_tric = (rows_I, cols_J, contx_K)
                num_cols_tric = [i for i, elem in enumerate(cols_J) if elem in self._num_features_idx]
                symb_cols_tric = [i for i, elem in enumerate(cols_J) if elem in self._symb_features_idx]
                c_tric = np.transpose(TriHSPAM.get_subcube_special(self._data, rows_I, cols_J, contx_K), (2,1,0))
            else:
                for jk in cols_contx_JK:
                    j_k_split = jk.split("_")
                    j = int(j_k_split[0].replace("f",""))
                    k = int(j_k_split[1])
                    if j not in cols_J:
                        cols_J.append(j)
                        cols_J.sort()
                    if k not in contx_K:
                        contx_K.append(k)
                        contx_K.sort()
                rows_I.sort()
                cols_J.sort()
                contx_K.sort()
                candidate_tric = (rows_I, cols_J, contx_K)
                num_cols_tric = [i for i, elem in enumerate(cols_J) if elem in self._num_features_idx]
                symb_cols_tric = [i for i, elem in enumerate(cols_J) if elem in self._symb_features_idx]
                c_tric = np.transpose(TriHSPAM.get_subcube(self._data, rows_I, cols_J, contx_K), (2,1,0))
            rows_size, cols_size = len(rows_I), len(cols_J)
            conts_size = len(contx_K) if not self._relaxed else len(list(contx_K.values())[0])
            if rows_size >= self._min_I and cols_size >= self._min_J and conts_size >= self._min_K:
                fitness_val = h_var3(c_tric, num_cols_tric, symb_cols_tric)
                fit_val = fitness_val
                if fit_val <= self._coer_thresh:
                    has_sym = any(element in self._symb_features_idx for element in cols_J)
                    has_num = any(element in self._num_features_idx for element in cols_J)
                    if has_sym and has_num:
                        self._triclusters_type.setdefault("MixedTriclusters", []).append(candidate_tric)
                    elif has_num:
                        self._triclusters_type.setdefault("NumericTriclusters", []).append(candidate_tric)
                    elif has_sym:
                        self._triclusters_type.setdefault("SymbolicTriclusters", []).append(candidate_tric)
    
        self._triclusters = [item for tr_list in self._triclusters_type.values() for item in tr_list]
        return self
    
    def triclusters_(self):
        return self._triclusters

    def get_indices(self, i):
        return self._triclusters[i]
    
    def get_shape(self, i):
        x,y,z = self.get_indices(i)
        return (len(x), len(y), len(z))
    
    def get_tricluster(self, i):
        X,Y,Z = self._triclusters[i]
        tricluster = TriHSPAM.get_subcube(self._data, X, Y, Z) if not self._relaxed else TriHSPAM.get_subcube_special(self._data, X, Y, Z)
        return tricluster

    def save_triclusters(self, time_as_cols, file_path):
        for i in range(len(self._triclusters)):
            tricluster = self.get_tricluster(i)
            if not time_as_cols:
                tricluster = np.transpose(tricluster, (2,1,0))
            utils.write_3d_array_to_txt(tricluster, file_path)

    def filter(self, overlapping_threshold):
        filtered_trics = list()
        for t1 in self._triclusters:
            for t2 in self._triclusters[1:]:
                i = set.intersection(t1[0], t2[0])
                j = set.intersection(t1[1], t2[1])
                k = set.intersection(t1[2], t2[2])
                if len(i) > 0 and len(j) > 0 and len(k) > 0:
                    t1_size = TriHSPAM.tric_size(t1)
                    t2_size = TriHSPAM.tric_size(t2)
                    overl_area = TriHSPAM.tric_size((i,j,k))
                    perc_overlp = overl_area / (t1_size + t2_size - overl_area)
                    if perc_overlp >= overlapping_threshold:
                        filtered_trics.append(max(t1, t2, key=(lambda x: TriHSPAM.tric_size(x))))

        self._triclusters = filtered_trics
        return self._triclusters

    def merge(self, overlapping_threshold):
        merged_triclusters = list()
        for t1 in self._triclusters:
            for t2 in self._triclusters[1:]:
                i = set.intersection(t1[0], t2[0])
                j = set.intersection(t1[1], t2[1])
                k = set.intersection(t1[2], t2[2])
                if len(i) > 0 and len(j) > 0 and len(k) > 0:
                    t1_size = TriHSPAM.tric_size(t1)
                    t2_size = TriHSPAM.tric_size(t2)
                    overl_area = TriHSPAM.tric_size((i,j,k))
                    perc_overlp = overl_area / (t1_size + t2_size - overl_area)
                    if perc_overlp >= overlapping_threshold:
                        u_i = set.union(t1[0], t2[0])
                        u_j = set.union(t1[1], t2[1])
                        u_k = set.union(t1[2], t2[2])
                        merged_triclusters.append((u_i, u_j, u_k))
        self._triclusters = merged_triclusters
        return self._triclusters

    @staticmethod
    def tric_size(tric):
        return len(tric[0]) * len(tric[1]) * len(tric[2])
    
    # @staticmethod
    # def get_subcube(data, X_rows, J_cols, Z_conts):

    #     X,Y,Z = X_rows, J_cols, Z_conts
    #     subcube = np.zeros(data.shape, dtype=object) 
    #     for z_k, x_i, y_j in list(product(Z, X, Y)):
    #             subcube[y_j, x_i, z_k] = data[y_j, x_i, z_k]

    #     non_zero_indices_axis0 = np.any(subcube != 0, axis=(1, 2))
    #     non_zero_indices_axis1 = np.any(subcube != 0, axis=(0, 2))
    #     non_zero_indices_axis2 = np.any(subcube != 0, axis=(0, 1))

    #     # Extract the submatrix without zeros
    #     subcube = subcube[non_zero_indices_axis0, :, :]
    #     subcube = subcube[:, non_zero_indices_axis1, :]
    #     subcube = subcube[:, :, non_zero_indices_axis2]
    #     return subcube

    @staticmethod
    def get_subcube(data, X_rows, J_cols, Z_conts):
        subcube_sizes = (len(J_cols), len(X_rows), len(Z_conts))
        subcube = np.zeros(subcube_sizes, dtype=object) 
        for i, x_i in enumerate(X_rows):
            for j, y_j in enumerate(J_cols):
                for k, z_k in enumerate(Z_conts):
                    subcube[j, i, k] = data[y_j, x_i, z_k]
        return subcube
    
    @staticmethod
    def get_subcube_special(data, X_rows, J_cols, Z_conts):
        subcube_sizes = (len(J_cols), len(X_rows), len(list(Z_conts.values())[0]))
        subcube = np.zeros(subcube_sizes, dtype=object) 
        for i, x_i in enumerate(X_rows):
            for j, y_j in enumerate(J_cols):
                for k, z_k in enumerate(Z_conts[x_i]):
                    subcube[j, i, k] = data[y_j, x_i, z_k]
        return subcube