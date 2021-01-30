from scipy.io import wavfile
from scipy import signal
from python_speech_features import mfcc
from hmmlearn import hmm
import numpy as np
import joblib
import librosa
import soundfile as sf
import os

def LabelTransfer(label):
    text2num = {
        'irport' : 0,
        'bus' : 1,
        'metro_station' : 2,
        'metro' : 3,
        'park' : 4,
        'public_square' : 5,
        'shopping_mall' : 6,
        'street_pedestrian' : 7,
        'street_traffic' : 8,
        'tram' : 9
    }
    num2text = {
        0 : 'irport',
        1 : 'bus',
        2 : 'metro_station',
        3 : 'metro',
        4 : 'park',
        5 : 'public_square',
        6 : 'shopping_mall',
        7 : 'street_pedestrian',
        8 : 'street_traffic',
        9 : 'tram'
    }
    if type(label) == str:
        return text2num.get(label)
    elif type(label) == int:
        return num2text.get(label)
    else:
        return None

def gen_wavlist(wavpath):
    wavdict = {}
    labeldict = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.sep.join([dirpath,filename])
                fileid = filename.strip('.wav')
                wavdict[fileid] = filepath
                label = fileid.split('-')[0]
                label = LabelTransfer(label)
                labeldict[fileid] = label
    return wavdict, labeldict

def compute_mfcc(file):
	audio, fs = librosa.load(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26, nfft=551)
	return mfcc_feat

class Model():
    def __init__(self, CATEGORY=None, n_comp=3, n_mix=3, cov_type='diag', n_iter=1000):
        super(Model, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY) 
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
        for k in range(self.category):
            model = hmm.GMMHMM(n_components = self.n_comp, n_mix = self.n_mix, covariance_type = self.cov_type, n_iter = self.n_iter)
            self.models.append(model)
    
    def train(self, wavdict=None, labeldict=None):
        for k in range(self.category):
            model = self.models[k]
            for x in wavdict:
                if labeldict[x] == self.CATEGORY[k]:
                    mfcc_feat = compute_mfcc(wavdict[x])
                    model.fit(mfcc_feat)
    
    def test(self, wavdict=None, labeldict=None):
        result = []
        for k in range(self.category):
            subre = []
            label = []
            model = self.models[k]
            for x in wavdict:
                mfcc_feat = compute_mfcc(wavdict[x])
                re = model.score(mfcc_feat)
                subre.append(re)
                label.append(labeldict[x])
            result.append(subre)
        result = np.vstack(result).argmax(axis=0)
        result = [self.CATEGORY[label] for label in result]
        print('分类得到结果: ',result)
        print('原始标签类别: ',label)
        totalnum = len(label)
        correnum = 0
        for i in range(totalnum):
            if result[i] == label[i]:
                correnum += 1
        print('分类正确率: ',correnum/totalnum)

    def save(self, path="mymodels.pkl"):
        joblib.dump(self.models, path)

    def load(self, path="mymodels.pkl"):
        self.models = joblib.load(path) 



if __name__ == "__main__":
    CATEGORY = [0,1,2,3,4,5,6,7,8,9]
    wavdict, labeldict = gen_wavlist('data') #改为训练集的路径
    print(wavdict)
    print(labeldict)
    testdict, testlabel = gen_wavlist('data_test') #改为测试集的路径
    print("----------------data loading finish!---------------")
    models = Model(CATEGORY=CATEGORY)
    models.train(wavdict=wavdict, labeldict=labeldict)
    print("---------------model training finish---------------")
    models.save()
    models.load()
    models.test(wavdict=testdict, labeldict=testlabel)
    print("---------------model testing finish----------------")

