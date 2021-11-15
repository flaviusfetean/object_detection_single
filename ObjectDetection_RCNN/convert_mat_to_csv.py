import scipy.io
import pandas as pd

mat_file = 'D:\\Deep_Learning_Projects\\datasets\\Caltech_101\\Annotations\\Airplanes\\annotation_0001.mat'

mat = scipy.io.loadmat(mat_file)
mat = {k: v for k, v in mat.items() if k[0] != '_'}

# parsing arrays in arrays in mat file
data = {}
for k, v in mat.items():
    arr = v[0]
    for i in range(len(arr)):
        sub_arr = v[0][i]
        lst = []
        for sub_index in range(len(sub_arr)):
            vals = sub_arr[sub_index][0][0]
            lst.append(vals)
        data['row_{}'.format(i)] = lst

data_file = pd.DataFrame.from_dict(data, orient='index', columns=['fname', 'x1', 'y1', 'x1', 'y2', 'label'])
data_file.to_csv("sample_annotation.csv")
print("DONE")