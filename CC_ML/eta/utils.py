import math
import numpy as np
from obspy import read, core
import pickle
from scipy.fft import rfft, rfftfreq

def construct_vectors_to_compare(sensors, trace_num):
    four_days = make_vector_dict(0, 96*3600.0, sensors, trace_num)

    two_days = []
    time_frame = 3600*48
    for i in range(0, 24*3601, 24*3600):
        two_days.append(make_vector_dict(i, time_frame*1.0, sensors, trace_num))

    one_day = []
    time_frame = 3600*24
    for i in range(0, 48*3600, 12*3600):
        one_day.append(make_vector_dict(i, time_frame*1.0, sensors, trace_num))

    return four_days, two_days, one_day

def make_vector_dict(start_time, time_frame, sensors, trace_num):
    start = core.UTCDateTime("2017-07-01T00:00:00.000000Z") + start_time
    data = read_and_fft(sensors, time_frame, start_time, trace_num)
    retVal = dict()
    for key, val in data.items():
        for i in range(25, 61, 5):
            frequency_index = np.argmin(np.abs(val['freq'] - i))
            if i in retVal:
                if len(retVal[i][0]) == 1:
                    retVal[i][0].append(val['amp'][frequency_index])
                    retVal[i][0] = get_normal_complex(retVal[i][0])
                    retVal[i].append([val['amp'][frequency_index]])
                else:
                    retVal[i][1].append([val['amp'][frequency_index]])
                    retVal[i][1] = get_normal_complex(retVal[i][1])
            else:
                retVal[i] = [[val['amp'][frequency_index]]]

    return retVal

def complex_dot_product(a, b):
    assert(len(a) == len(b))
    dot_product = 0
    for i in range(len(a)):
        dot_product += np.conj(a[i]) * b[i]
    
    return dot_product

def magnitude(a):
    magnitude = 0
    for dimension in a:
        magnitude += dimension ** 2
    magnitude = magnitude ** (1/2)
    return magnitude

def complex_magnitude(a):
    magnitude = 0
    for dimension in a:
        magnitude += dimension * np.conj(dimension)
    magnitude = magnitude ** (1/2)
    return magnitude

def complex_conjugate(complex):
    x = np.real(complex)
    y = np.imag(complex)
    conjugate = x + y * 1j
    return conjugate

def get_normal_complex(a):
    normal = []
    mag = complex_magnitude(a)
    for dimension in a:
        normal.append(dimension / mag)
    normal = np.array(normal)
    return normal

def get_normal(a):
    normal = []
    mag = magnitude(a)
    for dimension in a:
        normal.append(dimension / mag)
    normal = np.array(normal)
    return normal

def copyof(list):
    deep_copy = []
    for item in list:
        deep_copy.append(item)
    return deep_copy

def make_freq_to_vector(vector_with_file_path, min_freq, max_freq, step):
    freq_to_vectors = dict()
    forest_data = read_and_fft(vector_with_file_path, 24*60*60, 0, 0)
    for freq in range (min_freq, max_freq+1, step):
        cur_t_vec = copyof(vector_with_file_path)
        if freq == 0:
            freq = 1
        for j, sensor in enumerate(cur_t_vec):
            target_index = np.argmin(np.abs(forest_data[sensor]['freq']-freq))
            try:
                cur_t_vec[j] = forest_data[sensor]['amp'][target_index]
            except:
                print(f"{sensor} appears to not exist")
        freq_to_vectors[freq] = cur_t_vec
    return freq_to_vectors

def angle_between_complex(a, b):

    a_real, a_imag = a.real, a.imag
    b_real, b_imag = b.real, b.imag

    dot_product = a_real * b_real + a_imag * b_imag

    magnitude_a = math.sqrt(a_real**2 + a_imag**2)
    magnitude_b = math.sqrt(b_real**2 + b_imag**2)

    cos_angle = dot_product / (magnitude_a * magnitude_b)

    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle = math.acos(cos_angle)
    return angle


def read_and_fft(files, time_frame, start_time, trace_num):
    assert (trace_num == 0 or trace_num == 1 or trace_num == 2)
    file_to_frequencies = dict()

    for file in files:
        try:
            data = read(file)
            print(file)
        except:
            print(f"Failed to read {file} Skipping to next file.")
            continue
        try:
            start = data[trace_num].meta['starttime'] + start_time
            end = start + time_frame
            delta = data[trace_num].meta['delta']
            sample_rate = data[trace_num].meta['sampling_rate']
            
        except:
            print(f"Failed to select trace from {file}. Skipping to next file.")
            print(f"Actual trace len = {len(data)}, expected len = 3.")
            continue

        
        pre_trim_time = float(data[trace_num].meta['endtime'] - data[trace_num].meta['starttime'])
        if (pre_trim_time > time_frame):
            data.trim(start, end)
        else:
            time_frame = pre_trim_time
        print("len(data):", len(data))
        data = np.int64(data[trace_num].data)
        amplitudes = rfft(data)
        frequencies = rfftfreq(int(time_frame * sample_rate), delta)
        amp_and_freq = dict()

        amp_and_freq['amp'] = amplitudes
        amp_and_freq['freq'] = frequencies

        file_to_frequencies[file] = amp_and_freq
    
    return file_to_frequencies

def store_data_in_pickle(data):
    with open("sensor_data_frequency_space.pickle", "wb") as file:
        pickle.dump(data, file)

def compare_signals_through_space(data, target_frequency=40):
    if type(data) == str:
        with open(data, "rb") as file:
            data = pickle.load(file)

    vectors = []
    vec1 = []
    vec2 = []
    for sensor, data in data.items():
        target_index = np.argmin(np.abs(data['freq'] - target_frequency))
        if len(vec1) == 0:
            vec1.append(data['amp'][target_index])
        elif len(vec1) == 1:
            vec1.append(data['amp'][target_index])
            vec2.append(data['amp'][target_index])
        else:
            vectors.append(get_normal_complex(vec1))
            vec1 = [data['amp'][target_index]]
            vec2.append(data['amp'][target_index])
            vectors.append(get_normal_complex(vec2))
            vec2 = []

    if len(vec1) == 2:
        vectors.append(get_normal_complex(vec1))

    vectors = np.array(vectors)
    
    return vectors

def euclidean_distance(x, y):
    distance = 0
    for xi, yi in zip(x, y):
        cur = xi - yi
        cur = cur ** 2
        distance += cur
    distance = distance ** 1/2
    return distance





def comparision_of_geometric_phase_from_different_spans(sensors, trace_num):
    print(f"using sensors: {sensors}")
    two, one, half = construct_vectors_to_compare(sensors, trace_num)
    print(f"using two")
    thetas_two = dict()
    for key, val in two.items():
        for i in range(1, len(val)):
            thetas_two[key] = (angle_between_complex_vectors(val[i-1], val[i]))
    print(thetas_two)  

    print(f"using one")
    thetas_array_one = []
    for vector_dict in one:
        thetas_one = dict()
        for key, val in vector_dict.items():
            for i in range(1, len(val)):
                thetas_one[key] = (angle_between_complex_vectors(val[i-1], val[i]))
        thetas_array_one.append(thetas_one)

    for th1 in thetas_array_one:
        print(th1)

    print(f"using half")
    thetas_array_half = []
    for vector_dict in half:
        thetas_half = dict()
        for key, val in vector_dict.items():
            for i in range(1, len(val)):
                thetas_half[key] = (angle_between_complex_vectors(val[i-1], val[i]))
        thetas_array_half.append(thetas_half)
    for thhalf in thetas_array_half:
        print(thhalf)
    return thetas_array_half

        
def angle_between_complex_vectors(a, b):
    assert (0.999999 < complex_magnitude(a) and complex_magnitude(a) < 1.000001)
    assert (0.999999 < complex_magnitude(b) and complex_magnitude(b) < 1.000001)

    dot_prod = complex_dot_product(a, b)

    ratio = np.imag(dot_prod) / np.real(dot_prod)
    theta = math.atan(ratio)
    if np.real(dot_prod) < 0:
        theta += math.pi

    return theta   

def angle_between_complex_vectors_acos(a, b):
    dot = complex_dot_product(a, b)
    theta = np.sign(np.imag(dot)) * math.acos(np.real(dot))
    return theta

def angle_between_complex_vectors_acos_euclidian(a, b):
    dot = complex_dot_product(a, b)
    theta = math.acos(np.real(dot))
    return theta


def angle_between_complex_vectors_atan2(a, b):
    dot_prod = complex_dot_product(a, b)

    theta = math.atan2(np.imag(dot_prod), np.real(dot_prod))
    return theta
