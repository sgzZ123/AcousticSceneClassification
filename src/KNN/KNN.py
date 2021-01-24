from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import wavio
import csv
import os
from tqdm import tqdm
import sys

from get_feature import get_stft_amp, get_mel_amp

def LabelTransfer(label):
    text2num = {
        'airport': 0,
        'bus': 1,
        'metro_station': 2,
        'metro': 3,
        'park': 4,
        'public_square': 5,
        'shopping_mall': 6,
        'street_pedestrian': 7,
        'street_traffic': 8,
        'tram': 9
    }
    num2text = {
        0: 'airport',
        1: 'bus',
        2: 'metro_station',
        3: 'metro',
        4: 'park',
        5: 'public_square',
        6: 'shopping_mall',
        7: 'street_pedestrian',
        8: 'street_traffic',
        9: 'tram'
    }
    if type(label) == str:
        return text2num.get(label)
    elif type(label) == int:
        return num2text.get(label)
    else:
        return None


def kNN_Classify(inX, dataSet, labels, k):

    #inXΪ����������dadaSetΪ�ܵ�ѵ������labelsΪѵ������Ӧ�ķ��࣬k����ȡ20
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5  #�����
    sortedDistIndicies = distances.argsort()
    #argsort�������ص�������ֵ��С���������ֵ
    classCount = {} # ����һ���ֵ�
    #ѡ��k�������
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        #����k��������и������ֵĴ���
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
 
    # ���س��ִ�����������ǩ
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex   #����inX��Ӧ�ķ���

class Solver(object):
    def __init__(self, params) -> None:
        super(Solver, self).__init__()
        self.params = params

    def load_data(self, params, dataType):
        files = []
        with open(os.path.join(params['data_path'], 'evaluation_setup/fold1_{}.csv'.format(dataType)), 'r') as f:
            r = csv.reader(f)
            for row in r:
                files.append(row[0])
        files = files[1:]
        print('{} files found'.format(len(files)))

        feature_fn = None
        data = None
        label = np.zeros(len(files))
        if params['feature_type'] == 'mel':
            data = np.zeros((len(files), 128 * 499))
            feature_fn = get_mel_amp
        elif params['feature_type'] == 'stft':
            data = np.zeros((len(files), 500 * 1001))
            feature_fn = get_stft_amp
        else:
            print('feature type unrecognized!')
            raise ValueError

        print('loading data')
        for i in tqdm(range(len(files))):
            if dataType != 'test':
                f = files[i].split('\t')[0]
            label[i] = LabelTransfer(f[6:].split('-')[0])
            f = os.path.join(params['data_path'], f)
            _data, _fs = sf.read(f)
            data[i] = feature_fn(_data, _fs).flatten()
        
        print(sys.getsizeof(data) / 1024 / 1024 / 1024)
        
        return data, label

    def evaluate(self):
        data_train, label_train=self.load_data(self.params, 'train')
        data_test, label_test = self.load_data(self.params, 'evaluate')
        result= np.zeros(label_test.shape[0])
        for i in range(label_test.shape[0]):
            result[i]=kNN_Classify(data_test[i], data_train, label_train, 20)

        print('{:.3f} accuracy'.format((result==label_test).sum()/label_test.shape[0]*100))


if __name__ == '__main__':
    params = {
        'data_path': '/home/v-yuez1/classification/data/train',
        'feature_type': 'mel',
    }
    solver = Solver(params)
    solver.evaluate()
