from __future__ import print_function

import cPickle as pickle
import warnings

import numpy as np
from sklearn import metrics

from datasets import DatasetBuilder


def patient_id(patient, station):
    station_data = {
        1: {
            1: '63162_1',
            2: '63162_2',
            3: '63162_3',
            4: '63162_4'
        },
        2: {
            1: '70020_1',
            2: '70020_2',
            3: '70020_3',
            4: '70020_4'
        },
        3: {
            1: '69871_1',
            2: '69871_2',
            3: '56210_3',
            4: '56210_4'
        }
    }
    return station_data[patient][station]


patient = 2
station = 1

network = 'UNet'
name = 'unet-test_64x64x3_64x64x1_10_31_{s}'
y_pred_path = '/home/jamesneill/vessel_seg/session/{network}/results/{name}.pkl'
y_true_path = '/home/jamesneill/dataset/cleaned_gt/cleaned_gt_{patient}.tiff'
warnings.simplefilter('ignore', UserWarning)

scores = []

for station in range(3,5):

    print('{b} {s} {b}'.format(b='='*15, s=station))
    y_true = y_true_path.format(patient=patient_id(patient, station))
    y_pred = y_pred_path.format(network=network, name=name.format(s=station))


    y_true = DatasetBuilder.import_tiff(y_true).flatten()
    y_true[y_true < 0.5] = 0
    y_true[y_true >= 0.5] = 1

    y_pred = pickle.load(open(y_pred, 'rb')).flatten()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1


    # intersection = np.sum(y_pred[y_true == 1])
    # union = np.sum(y_pred) + np.sum(y_true)
    dsc_metric = metrics.f1_score(y_pred, y_true)
    scores.append(dsc_metric)

    # dsc_coords[station - 1] += '({s}, {d})'.format(s=step, d=dsc_metric)
    # union_coords[station - 1] += '({s}, {u})'.format(s=step, u=union)
    # intersection_coords[station - 1] += '({s}, {i})'.format(s=step, i=intersection)

    print('Station #{} --> DSC  {:0.3f}'.format(station, dsc_metric))
    # print('            --> INT  {}'.format(intersection))
    # print('            --> UNI  {}'.format(union))
print('Average --> {:0.3f}'.format(np.average(scores)))

#
# for index, _ in enumerate(dsc_coords):
#     print('Station #{}'.format(index + 1))
#     print('DSC', dsc_coords[index])
#     print('UNION', union_coords[index])
#     print('INTERSECTION', intersection_coords[index])
