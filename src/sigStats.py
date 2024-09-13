import math, sys, numpy as np, decimal as dc
from scipy.stats import mode
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import LabelEncoder

class NullModel:
    
    def __init__(self, data, Y_cat, coherence, uniform):
        __, m, self.ntime = data.shape
        self.coherence = coherence
        self.uniform = uniform
        self.Y_cat = Y_cat
        if not(uniform): 
            self.empirical, self.mins, self.maxs = [], [], []
            for j in range(m):
                values = np.reshape(data[:,j,:],-1)
                self.mins.append(np.min(values))
                self.maxs.append(np.max(values))
                if j in Y_cat:
                    unique, counts = np.unique(values, return_counts=True)
                    self.empirical.append(counts/len(values))
                else: self.empirical.append(ECDF(values))

        
    def pattern_prob(self, tric):
        if self.uniform: 
            p = self.coherence**(tric.m*tric.t)
        else: 
            p = 1
            for j, var in enumerate(tric.J):
                for k in range(tric.t):
                    value = tric.pattern[j,k]
                    if var in self.Y_cat:
                        p *= self.empirical[var][int(value)]
                    else:
                        delta = (self.maxs[var]-self.mins[var])*self.coherence/2
                        lower, upper = max(value-delta,self.mins[var]), min(value+delta,self.maxs[var])
                        p *= self.empirical[var](upper)-self.empirical[var](lower)
                        
        return p*(self.ntime - tric.t + 1)

class Tricluster:
    
    def __init__(self, data, Y_cat, I, J, K):
        for j,var in enumerate(range(data.shape[1])):
            if var in Y_cat: 
                data[:,j,:] = encode_2d_array(data[:,j,:])
            else:
                data[:,j,:] = data[:,j,:].astype(float)

        self.I, self.J, self.K = I, J, K
        if isinstance(K, dict):
            subspace = Tricluster.get_subcube_special(data, I, J, K)
        else:
            subspace = Tricluster.get_subcube(data, I, J, K)
        self.n, self.m, self.t = subspace.shape
        self.pattern = np.array([]).reshape(0,self.t)
        for j,var in enumerate(J):
            if var in Y_cat: central = mode(subspace[:,j,:], keepdims=True, axis=0, nan_policy='omit')[0]
            else: central = np.array([np.nanmean(subspace[:,j,:], axis=0)])
            self.pattern = np.concatenate((self.pattern, central), axis=0)
    
    
    @staticmethod
    def get_subcube(data, X_rows, J_cols, Z_conts):
        subcube_sizes = (len(X_rows), len(J_cols), len(Z_conts))
        subcube = np.zeros(subcube_sizes, dtype=object) 
        for i, x_i in enumerate(X_rows):
            for j, y_j in enumerate(J_cols):
                for k, z_k in enumerate(Z_conts):
                    subcube[i, j, k] = data[x_i, y_j, z_k]
        return subcube
    
    @staticmethod
    def get_subcube_special(data, X_rows, J_cols, Z_conts):
        subcube_sizes = (len(X_rows), len(J_cols), len(list(Z_conts.values())[0]))
        subcube = np.zeros(subcube_sizes, dtype=object) 
        for i, x_i in enumerate(X_rows):
            for j, y_j in enumerate(J_cols):
                for k, z_k in enumerate(Z_conts[x_i]):
                    subcube[i, j, k] = data[x_i, y_j, z_k]
        return subcube
    

def bin_prob(n, p, k):
    if p==1: return 1
    ctx = dc.Context()
    arr = math.factorial(n) // math.factorial(k) // math.factorial(n-k)
    bp = (dc.Decimal(arr) * ctx.power(dc.Decimal(p), dc.Decimal(k)) * ctx.power(dc.Decimal(1-p), dc.Decimal(n-k)))
    return float(bp) if sys.float_info.min < bp else sys.float_info.min

def significance(data, Y_cat, trics, coherence, uniform=False, verbose=False):

    model = NullModel(data, Y_cat, coherence, uniform)
    
    pvalues = []
    n, m, t = data.shape
    for tric in trics:
        pvalue = 0
        p = model.pattern_prob(tric)
        for i in range(tric.n, n+1):
            pvalue += bin_prob(n, p, i)
        pvalues.append(pvalue)
        if verbose:
            print("p_value = %E (p_pattern = %f)"%(pvalue,p))
    return pvalues


def encode_2d_array(data):
    """
    Encode categorical variables in a 2D NumPy array.

    Parameters:
    data (numpy.ndarray): Input 2D array with categorical variables.

    Returns:
    numpy.ndarray: Encoded 2D array.
    """
    label_encoder = LabelEncoder()
    encoded_data = np.empty(data.shape, dtype=np.int32)

    for i in range(data.shape[1]):
        encoded_data[:, i] = label_encoder.fit_transform(data[:, i])

    return encoded_data