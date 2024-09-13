from collections import Counter
import numpy as np

def compute_representative_patterns(tricluster, mode_feats, mean_features, median_features):
    """
    Format: [[], []]
    """
    def compute_column_means_medians(arr, mean=True):
        # Initialize an array to store the means
        vals = np.zeros(arr.shape[1])
        
        # Iterate over each column
        for i in range(arr.shape[1]):
            column_data = arr[:, i]
            
            # Convert non-numeric strings to NaN
            converted_column = []
            for val in column_data:
                try:
                    converted_column.append(float(val))
                except ValueError:
                    converted_column.append(np.nan)
            
            # Compute mean, ignoring NaN values
            if mean:
                vals[i] = round(np.nanmean(converted_column),2)
            else:
                vals[i] = round(np.nanmedian(converted_column),2)
        
        return vals
    def compute_column_modes(arr):
        # Initialize an array to store the modes
        modes = []

        # Iterate over each column
        for i in range(arr.shape[1]):
            column_data = arr[:, i]
            
            # Count occurrences of each unique string
            counter = Counter(column_data)
            
            # Get the string with the highest count (mode)
            mode = max(counter, key=counter.get)
            modes.append(mode)
        
        return modes
        
    tricluster = np.transpose(tricluster, (2,1,0))
    all_features = mode_feats + mean_features + median_features
    all_features.sort()
    mean_features_idx = [i for i in range(len(all_features)) if all_features[i] in mean_features]
    median_features_idx = [i for i in range(len(all_features)) if all_features[i] in median_features]
    mode_features_idx = [i for i in range(len(all_features)) if all_features[i] in mode_feats]
    all_patterns = []
    for slice_idx in range(tricluster.shape[0]):
        slice_data = tricluster[slice_idx]
        slice_means = compute_column_means_medians(slice_data)
        slice_medians = compute_column_means_medians(slice_data, mean=False)
        slice_modes = compute_column_modes(slice_data)
       
        pattern = []
        for f in range(len(all_features)):
            if f in mean_features_idx:
                pattern.append(slice_means[f]) 
            elif f in mode_features_idx:
                pattern.append(slice_modes[f])
            elif f in median_features_idx:
                pattern.append(slice_medians[f])
        all_patterns.append(pattern)
    return all_patterns
