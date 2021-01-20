from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import wavio
import csv

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from get_feature import get_stft_amp, get_mel_amp


wav_file_path_all=[]
Y=[]  #label
files=[]
with open('E:\大四上/信号建模与算法实践/大实验/test.csv', 'r') as f: #请改路径
    r = csv.reader(f)
    for row in r:
        files.append(row[0])

files = files[1:]
X = np.zeros(shape=(len(files),128,499))		#创建data的空矩阵，mel为(128,499)，sift为(500, 1001)
for f in files:
	wav_file_path_all.append('E:\大四上/信号建模与算法实践/大实验/'+f.split('.')[0]+'.wav') #.wav文件存放的绝对地址
	Y.append(f.split('.')[1])#Y为'wavairport'这样，不过不影响分类
	
for wav_file_path in wav_file_path_all:
	_data,_fs = sf.read(wav_file_path)
	#sift是下面这些
	# stft_amp = get_stft_amp(_data,_fs)                      #stft_amp.shape (500, 1001)
	# X[wav_file_path_all.index(wav_file_path)]=stft_amp
	
	#mel是下面这些
	wav_data=wavio.read(wav_file_path)
	data=wav_data.data.astype(float)/np.power(2,wav_data.sampwidth*8-1)
	data=np.asarray(data)
	mel_amp=get_mel_amp(data[:,0],fs=_fs)                         #mel_amp.shape (128,499)
	X[wav_file_path_all.index(wav_file_path)]=mel_amp  #填入data矩阵


# 真正训练就这两行，make_pipeline是并行计算，StandardScaler()是数据预处理函数，SVC是分类函数
# LinearSVC参数：
# C:float, default=1.0 --惩罚参数
# kernel：{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf' --核函数
# degree：int, default=3 --核函数为poly时多项式的维度
# gamma：{'scale', 'auto'} or float, default='scale' --rbf,poly和sigmoid的核函数参数
#--下面不重要--
# coef0：float, default=0.0 --核函数的常数项，对于poly和sigmoid有用
# shrinking：bool, default=True
# probability：bool, default=False 
# tol：float, default=1e-3  --停止训练的误差值大小
# cache_size：float, default=200  --核函数cache缓存大小，单位MB
# class_weight：dict or 'balanced', default=None  --权重，默认都为1，balanced按数量平均，本实验数据挺平均的，不需要balanced
# verbose：bool, default=False  --多进程中应为False
# max_iter：int, default=-1 --最大迭代次数
# decision_function_shape：{'ovo', 'ovr'}, default='ovr' --多分类方法，一对多或者一对一，一对一很慢且没有提升，很少用
# break_ties：bool, default=False
# random_state：int, RandomState instance or None, default=None --数据洗牌时的种子值


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, Y)   #训练！

#这里把数据集又预测了一遍，可以改为测试集，请抄你的 测试并且计算准确率 的代码
print(clf.predict(X))#会输出'wavbus','wavairport'这样的分类结果

#print(clf.score(X,Y)) #可以输出平均准确率