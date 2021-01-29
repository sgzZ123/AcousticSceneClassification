from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import wavio
import csv
import os
from tqdm import tqdm
import sys
from get_feature import get_stft_amp,get_mel_amp

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

class HMM:
    """
    A: 状态转移概率矩阵 (#states,#states)
    B: 符号发射概率矩阵 (#states,#samples)
    Pi: 初始状态概率向量 (#states,1)
    """
    def __init__(self,A,B,Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def _forward(self,samples):
        """
        前向算法
        给定HMM模型参数A/B/Pi
        计算特定观测序列产生的概率
        """
        n_states = self.A.shape[0]
        n_samples = len(samples)
        F = np.zeros((n_states,n_samples))
        F[:,0] = self.Pi * self.B[:,samples[0]]
        for t in range(1,n_samples):
            for n in range(n_states):
                F[n,t] = np.dot(F[:,t-1],(self.A[:,n]))*self.B[n,samples[t]]
        return F

    def _backward(self,samples):
        """
        后向算法
        """
        n_states = self.A.shape[0]
        n_samples = len(samples)
        X = np.zeros((n_states,n_samples))
        X[:,n_samples-1] = 1
        for t in reversed(range(n_samples-1)):
            for n in range(n_states):
                X[n,t] = np.sum(X[:,t+1]*self.A[n,:]*self.B[:,samples[t+1]])
        return X
        
    def baum_welch_train(self,samples,criterion=0.1):
        """
        根据观测序列预测HMM模型参数
        """
        n_states = self.A.shape[0]
        n_samples = len(samples)
        done = False
        count = 0
        while (not done) and count < 10:
            alpha = self._forward(samples)
            beta = self._backward(samples)
            if count==0 :
                xi = np.zeros((n_states,n_states,n_samples-1))
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T,self.A)*self.B[:,samples[t+1]].T,beta[:,t+1])
                if denom ==0:
                    return
                for i in range(n_states):
                    numer = alpha[i,t]*self.A[i,:]*self.B[:,samples[t+1]].T*beta[:,t+1].T 
                    xi[i,:,t] = numer/denom
            gamma = np.sum(xi,axis=1)
            prod = (alpha[:,n_samples-1]*beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,prod/np.sum(prod)))
            newPi = gamma[:,0]
            newA = np.sum(xi,2)/np.sum(gamma[:,:-1],axis=1).reshape(-1,1)
            newB = np.copy(self.B)
            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(num_levels):
                mask = samples == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1)/sumgamma
            if np.max(abs(self.Pi-newPi))<criterion and np.max(abs(self.A-newA))<criterion and np.max(abs(self.B-newB))<criterion:
                done = 1
            else:
                t = 0
                i = 0
                count += 1
                self.A = newA
                self.B = newB
                self.Pi = newPi
        self.A = newA
        self.B = newB
        self.Pi = newPi

    def viterbi(self,samples):
        """
        维特比译码
        返回V preV
        矩阵V V[s][t]表示max P(o1,..,ot,q1,q2,...,qt-1,qt=s|Pi,A,B)
        矩阵preV: 指向t-1时刻的状态使得V[state][t]最大
        """
        n_states = self.A.shape[0]
        n_samples = len(samples)
        preV = np.zeros((n_samples-1,n_states),dtype=int)
        V = np.zeros((n_states,n_samples))
        V[:,0] = self.Pi*self.B[:,samples[0]]
        for t in range(1,n_samples):
            for n in range(n_states):
                seq_probs = V[:,t-1]*self.A[:,n]*self.B[n,samples[t]]
                preV[n_samples-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)
        return V,preV

    def build_viterbi_path(self,preV,last_state):
        """
        找到最优路径
        """
        T = len(preV)
        yield(last_state)
        for i in range(T-1,-1,-1):
            yield(preV[i,last_state])
            last_state = preV[i,last_state]
        
    def state_path(self,samples):
        """
        返回
        V[last_state,-1]: float 最优状态路径的概率
        path: list(int) 给定观测序列时的最优状态序列
        用于给定模型参数、观测序列时,得到其状态序列
        """
        V, preV = self.viterbi(samples)
        last_state = np.argmax(V[:,-1])
        path = list(self.build_viterbi_path(preV,last_state))
        path = list(reversed(path))
        return V[last_state,-1], path

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
        data = data.flatten()
        data = (-1)*data
        if params['feature_type'] == 'stft':
            data = data - 1
        elif params['feature_type'] == 'mel':
            data = data + 1
        print(sys.getsizeof(data) / 1024 / 1024 / 1024)
        return data, label
            
    def train(self):
        data, label = self.load_data(self.params, 'train')
        _samples = np.unique(data)
        print(_samples)
        n_samples = len(_samples)
        _label = np.unique(label)
        n_states = len(_label)
        A = np.random.dirichlet(np.ones(n_states),size=n_states)
        B = np.random.dirichlet(np.ones(n_samples),size=n_states)
        Pi = np.random.dirichlet(np.ones(1),size=n_states)
        Pi = Pi.flatten()
        self.hmm = HMM(A,B,Pi)
        self.hmm.baum_welch_train(data)
        print("traing finish")
        
    def evaluate(self):
        data, label = self.load_data(self.params, 'evaluate')
        corr = 0
        wron = 0
        for i in range(len(data)):
            v,path = self.hmm.state_path(data[i])
            res = max(path,key=path.count)
            if res == label[i]:
                corr += 1
            else:
                wron += 1
        print('accuracy',corr/(corr+wron))

if __name__ == '__main__':
    params = {
        'data_path': '/home/v-yuez1/classification/data/train',
        'feature_type': 'mel',
    }
    solver = Solver(params)
    solver.train()
    solver.evaluate()
            

            