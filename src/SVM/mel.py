from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import wavio
import csv
import os
from tqdm import tqdm
import sys

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


class Solver(object):
    def __init__(self, params) -> None:
        super(Solver, self).__init__()
        self.params = params
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

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
            
    def train(self):
        data, label = self.load_data(self.params, 'train')
        self.clf.fit(data, label)

    def evaluate(self):
        data, label = self.load_data(self.params, 'evaluate')
        result = self.clf.predict(data)
        print('{:.3f} accuracy'.format((result==label).sum()/label.shape[0]*100))


if __name__ == '__main__':
    params = {
        'data_path': '/home/v-yuez1/classification/data/train',
        'feature_type': 'mel',
    }
    solver = Solver(params)
    solver.train()
    solver.evaluate()
