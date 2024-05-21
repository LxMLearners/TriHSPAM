from collections import namedtuple
IntervalState = namedtuple("IntervalState", "G s e")
InstantState = namedtuple("InstantState", "G tp")
State = namedtuple("State", "F V")
import math
import numpy as np

def state_to_string(intstate):
    return f"{intstate.G.F}#{intstate.G.V}"

def get_abstraction(abstractions_tree, value):
    return abstractions_tree.find_interval(value)

def get_abstraction_bins(bins_dict, value):
    for k, vs in bins_dict.items():
        if value in vs:
            return k

def find_abstractions(feature, abstractions_dict):
    return abstractions_dict[feature]
   

def get_abstraction_symb(abstractions, value):
    for k, v in abstractions.items():
        if v == value:
            return k
        
def get_feature_type(features_type, f_label):
    for k,v in features_type.items():
        if f_label in v:
            return k

def data_to_instants(data_vector, feature_label, abstractions, symbolic):
    if not symbolic:
        data_vector = [float(x) for x in data_vector]
        data_vector = list(filter(lambda x: not np.isnan(x), data_vector))

    if len(data_vector) == 0:
        return []

    result = []
    if len(data_vector) > 1:
        for i in range(len(data_vector)):
            abst = get_abstraction_bins(abstractions, data_vector[i]) if not symbolic else get_abstraction_symb(abstractions, data_vector[i])
            if abst:
                state = State(f"{feature_label}_{i}", abst)
                new_int = InstantState(state, i)
                result.append(new_int)
    return result


def data_to_instant_mss(tw_data, feature_labels, features_type, abstractions):
    result = {}
    for x_i in range(len(tw_data[0])):
        for y_j in range(len(tw_data)):
            result.setdefault(f"X{x_i}", [])
            if get_feature_type(features_type, feature_labels[y_j]) == "numeric":
                next_seq = data_to_instants(
                    tw_data[y_j][x_i], feature_labels[y_j], find_abstractions(y_j, abstractions), False)
                result[f"X{x_i}"].extend(next_seq)
            else:
                next_seq = data_to_instants(
                    tw_data[y_j][x_i], feature_labels[y_j], find_abstractions(y_j, abstractions), True)
                result[f"X{x_i}"].extend(next_seq)

    return result



def restrictive_coocurences(mss, K):
    mss = sorted(mss, key=lambda x: x.tp)
    i = 0
    Z = list()
    while i < K:
        coocur = list(filter(lambda x: x.tp == i, mss))
        # if len(coocur) > 0:
        Z.append(coocur)
        i += 1
    return [list(map(lambda ist: state_to_string(ist), seq)) for seq in Z]



def seqs_to_spmf(seqs, output_file, time_constr=False):
    global mappings
    mappings = {}

    def put_map(v):
        try:
            next_k = sorted(list(mappings.values()))[-1]+1
        except:
            next_k = 0
        return mappings.setdefault(v, next_k)
    
    out_f = open(output_file, "w")
    for ss in seqs.values():
        l = ""
        t_i = 0
        # print(ss)
        for s in ss:
            if len(s) > 0:
                if time_constr:
                    l += f"<{t_i}> "
                    t_i += 1
                l += " ".join(list(map(lambda si: str(put_map(si)), s)))
                l += " -1 "
            else:
                if time_constr:
                    t_i += 1

        l += "-2"
        out_f.write(f"{l}\n")
        # print(l)


def get_sequences(sequences, feature_labels, abstractions):
    buffer = []
    itemOccurrenceCount = 0

    for i in range(len(sequences)):
        # buffer.append(str(i) + ":  ")

        sequence = sequences[i]
        startingANewItemset = True

        for token in sequence:
            try:
                token = int(token)
            except ValueError:
                continue
            if token >= 0:  # if it is an item
                if startingANewItemset:
                    startingANewItemset = False
                    buffer.append("(")
                else:
                    buffer.append(" ")

                mapping = {v: k for k, v in mappings.items()}
                buffer.append(mapping[token])
                itemOccurrenceCount += 1  # increase the number of item occurrences for statistics
            elif token == -1:  # if it is an itemset separator
                buffer.append(") ")
                startingANewItemset = True
            elif token == -2:  # if it is the end of the sequence
                break

        buffer.append("\n")

    result = "".join(buffer)
    return result



def get_subjects_patterns(file, feature_labels, abstractions):
    outfile = open(file, 'r')
    seqs = list()
    subj_ids = list()
    for line in outfile:
        seqs.append(line.split("#SUP: ")[0].rstrip().split(" "))
        subj_ids.append(line.split("#SUP: ")[
                        1].rstrip().split("#SID: ")[1].split(" "))
    seqs = get_sequences(seqs, feature_labels, abstractions).split("\n")
    return dict(zip(seqs, subj_ids))


def find_interval(int_to_find, list_intervals):
    for i in list_intervals:
        if i.G.F == int_to_find.F and i.G.V == int_to_find.V:
            return i


def parse_pattern_string(pattern_s):
    p_lst = pattern_s.split(" ")
    p_lst = list(map(lambda s: s.strip("()"), p_lst))
    return p_lst


def get_subject_intervals_pattern(pattern, subject_intervals):
    """
        pattern is a string representation of pattern
    """
    pattern_l = parse_pattern_string(pattern)
    final_intervals = []
    for si in subject_intervals:
        int_to_verify = f"{si.G.F}#{si.G.V}"
        if int_to_verify in pattern:
            final_intervals.append(si) 
    return final_intervals




def impute_missing_with_locf(data):
    """
    Impute missing values in a 2D NumPy array across rows using the 
    Last Observation Carried Forward (LOCF) method.

    Parameters:
    - data: np.array
        The input 2D data array with possibly missing values.

    Returns:
    - imputed_data: np.array
        The array with missing values imputed across rows using LOCF.
    """

    imputed_data = np.copy(data)

    # Iterate over rows
    for i in range(imputed_data.shape[0]):
        last_observation = None

        # Iterate over columns
        for j in range(imputed_data.shape[1]):
            if imputed_data[i, j] is None or (isinstance(imputed_data[i, j], float) and np.isnan(imputed_data[i, j])):
                # If the value is missing, use the last observed value (LOCF)
                imputed_data[i, j] = last_observation
            else:
                # Update last observed value
                last_observation = imputed_data[i, j]

    return imputed_data

def equal_width_binning_np(data, num_bins):
    """
    Perform equal-width binning on a 2D NumPy array over the entire matrix.

    Parameters:
    - data: np.array
        The input 2D data array.
    - num_bins: int
        The number of bins to create.

    Returns:
    - bins: dict
        Dictionary where keys are bin names and values are list with values.
    """

    # Check if num_bins is a valid value
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    # Flatten the 2D array
    flattened_data = data.flatten()
    mask = [not np.isnan(x) for x in flattened_data]
    flattened_data = flattened_data[mask]
    

    bins_equal_width = np.linspace(min(flattened_data), max(flattened_data), num_bins + 1)
    bin_indices_equal_width = np.digitize(flattened_data, bins_equal_width)

    bins = {}
    for v in range(len(flattened_data)):
        bin_i = bin_indices_equal_width[v]
        bins.setdefault(f"bin{bin_i}", [])
        if flattened_data[v] not in bins[f"bin{bin_i}"]:
            bins[f"bin{bin_i}"].append(flattened_data[v])

    return bins

def equal_freq_binning_np(data, num_bins):
    """
    Perform equal-freq binning on a 2D NumPy array over the entire matrix.

    Parameters:
    - data: np.array
        The input 2D data array.
    - num_bins: int
        The number of bins to create.

    Returns:
    - bins: dict
        Dictionary where keys are bin names and values are lists with values.
    """

    # Check if num_bins is a valid value
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    # Flatten the 2D array
    flattened_data = data.flatten()

    mask = [(not np.isnan(x)) for x in flattened_data]
    flattened_data = flattened_data[mask]

    # Calculate bin width
    bins_equal_frequency = np.percentile(flattened_data, np.linspace(0, 100, num_bins + 1))
    bin_indices_equal_frequency = np.digitize(flattened_data, bins_equal_frequency)   

    # Initialize the bins dictionary
    bins = {}
    for v in range(len(flattened_data)):
        bin_i = bin_indices_equal_frequency[v]
        bins.setdefault(f"bin{bin_i}", [])
        if flattened_data[v] not in bins[f"bin{bin_i}"]:
            bins[f"bin{bin_i}"].append(flattened_data[v])

    return bins

def unique_values_abstractions(data):
    """
    Create a dictionary with unique values from a 2D NumPy array of strings.

    Parameters:
    - data: np.array
        The input 3D array containing strings.

    Returns:
    - unique_values_dict: dict
        Dictionary with unique values as keys and values.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a 2D NumPy array of strings.")

    unique_values_dict = {}
    flattened_data = data.flatten()
    mask = [True if isinstance(x, str) else not math.isnan(x) for x in flattened_data]
    flattened_data = flattened_data[mask]

    for value in np.unique(flattened_data):
            if isinstance(value, str) or not math.isnan(value):
                unique_values_dict[value] = value

    return unique_values_dict

def extract_features(pattern_string):
    """
    Extract features from a string in the specified format.

    Parameters:
    - input_string: str
        The input string in the format "(f0_0#bin1 f1_0#y f2_0#bin2) (f0_1#bin1 f1_1#x f2_1#bin2) ...".

    Returns:
    - features_list: list
        List of extracted feature names.
    """
    substrings = pattern_string.split()

    substrings = [s.strip("()") for s in substrings]

    # Extract features from each substring
    features_list = [substring.split("#")[0] for substring in substrings]

    return features_list

def write_3d_array_to_txt(array, file_path):
    """
    Write the textual representation of a 3D NumPy array to a text file,
    printing each 2D matrix separately.

    Parameters:
    - array: 3D NumPy array
    - file_path: Path to the text file where the representation will be written
    """
    with open(file_path, 'a') as file:
        # Get the shape of the array
        shape = array.shape

        # Write array shape as a comment in the file
        file.write(f"# Array shape: {shape}\n\n")

        # Iterate through the 3D array and write each 2D matrix
        for i in range(shape[0]):
            file.write(f"# Matrix {i+1}\n")
            for j in range(shape[1]):
                for k in range(shape[2]):
                    file.write(f"{array[i, j, k]} ")
                file.write("\n")
            file.write("\n")

