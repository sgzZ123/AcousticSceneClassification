#This script is used to extract the spectrogram features and log mel features
from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import wavio


def get_stft_amp(wavdata,fs=44100):                         #extracting speech spectrogram features
	wind_dur=0.02                                          #window_len=20ms
	wind_shift=0.01                                        #window_shift=10ms
	dft_len=1000                                            #dft_len=900
	window_len=wind_dur*fs
	overlap=(wind_dur-wind_shift)*fs
	f,t,zxx=signal.stft(wavdata,1,window='hamming',nperseg=window_len,noverlap=overlap,nfft=dft_len,detrend=False,return_onesided=True)
	sub_freq=np.absolute(zxx[0:500,:])                     #the dimension of final feature:200 or 400
	return np.log10(sub_freq)

def get_mel_amp(wavdata,fs=44100):                         #extracting log mel spectrogram features
	wind_dur = 0.04
	wind_shift = 0.02
	window_len = int(wind_dur*fs)
	overlap=int((wind_dur-wind_shift)*fs)
	[f,t,X]=signal.spectral.spectrogram(wavdata,window='hamming',nperseg=window_len,noverlap=overlap,nfft=window_len,detrend=False,return_onesided=True,mode='magnitude')
	melW=librosa.filters.mel(sr=44100,n_fft=window_len,n_mels=128,fmin=0.,fmax=22100)
	melW /=  np.max(melW,axis=-1)[:,None] 
	melX = np.dot(melW,X)
	return np.log10(melX)

wav_file_path = 'audio/airport-lisbon-1000-40000-a.wav'                        #audio file

_data,_fs = sf.read(wav_file_path)

stft_amp = get_stft_amp(_data,_fs)                      #stft_amp.shape (500, 1001)
print(stft_amp.shape)
print('finish spectrogram')
#for wav in wav_list:
wav_data=wavio.read(wav_file_path)
data=wav_data.data.astype(float)/np.power(2,wav_data.sampwidth*8-1)
data=np.asarray(data)
mel_amp=get_mel_amp(data[:,0],fs=_fs)                         #mel_amp.shape (128,499)
print(mel_amp.shape)
print('finish log mel features')





