import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import os
import soundfile as sf
import wavio
import numpy as np
from tqdm import tqdm
import csv
import dcase_util
from dcase_util.containers import AudioContainer
from dcase_util.features import MelExtractor

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


class m_dataset(Dataset):
    def __init__(self, data_path, feature_type='stft', dataType='train') -> None:
        super(m_dataset, self).__init__()
        self.data = None
        self.label = None
        self.fs = None
        self.dataType=dataType
        self.feature_type = feature_type
        self._make_dataset(data_path, feature_type=feature_type)

    def _make_dataset(self, data_path, feature_type='stft'):
        files = []
        self.data = []
        self.label = []

        with open(os.path.join(data_path, 'evaluation_setup/fold1_{}.csv'.format(self.dataType)), 'r') as f:
            r = csv.reader(f)
            for row in r:
                files.append(row[0])

        files = files[1:]
        print('{} files found'.format(len(files)))

        for f in tqdm(files):
            if self.dataType != 'test':
                f = f.split('\t')[0]
            self.label.append(LabelTransfer(f[6:].split('-')[0]))
            f = os.path.join(data_path, f)
            _data, _fs = sf.read(f)
            if self.feature_type == 'stft':
                stft_amp = get_stft_amp(_data, fs=_fs)
                t = np.transpose(stft_amp)
            elif self.feature_type == 'mel':
                # mel_amp = get_mel_amp(_data, fs=_fs)
                # t = np.transpose(mel_amp)
                audio = dcase_util.containers.AudioContainer().load(filename=f,mono=True)
                mel_extractor = dcase_util.features.MelExtractor(n_mels=40, win_length_seconds=0.04, hop_length_seconds=0.02, fs=audio.fs)
                mel_data = mel_extractor.extract(y=audio)
                t = mel_data[:,:500]
            else:
                raise ValueError
            self.data.append(t)
            
        print('dataset initialized')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        t = self.data[index]
        return torch.tensor(t).float(), torch.tensor(self.label[index]).long()


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlock, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(256, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x) + x
        x = self.l3(x)
        return x
    

class ResLinearModel(nn.Module):
    def __init__(self, feature_type='stft', output_size=10):
        super(ResLinearModel, self).__init__()
        if feature_type == 'stft':
            input_size = 500*1001
        elif feature_type == 'mel':
            input_size = 128*499
        else:
            print('unable to find a proper feature type!')
            raise ValueError
        self.net1 = ResBlock(input_size, 256)
        self.net2 = ResBlock(256, 256)
        self.net3 = ResBlock(256, output_size)

        self.activate_fn = nn.Softmax()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.activate_fn(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, feature_type='stft', output_size=10):
        super(LinearModel, self).__init__()
        if feature_type == 'stft':
            input_size = 500*1001
        elif feature_type == 'mel':
            input_size = 128*499
        else:
            print('unable to find a proper feature type!')
            raise ValueError
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x


class TransModel(nn.Module):
    def __init__(self, feature_type='stft', output_size=10):
        super(TransModel, self).__init__()
        if feature_type == 'stft':
            d_model = 500
            self.output_layer = nn.Sequential(
                nn.Linear(1001*d_model, output_size), 
                nn.Sigmoid()
            )
        elif feature_type == 'mel':
            d_model = 128
            self.output_layer = nn.Sequential(
                nn.Linear(499*d_model, output_size), 
                nn.Sigmoid()
            )
        else:
            print('unable to find a proper feature type!')
            raise ValueError
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)


    def forward(self, x):
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, feature_type='stft', output_size=10):
        super(ConvModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.MaxPool2d(10, 62)
        )
        self.output = nn.Sequential(
            nn.Linear(64, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.net(x).squeeze(-1).squeeze(-1)
        print(x.shape)
        x = self.output(x)
        return x


class Solver(object):
    def __init__(self, params) -> None:
        super(Solver, self).__init__()
        self.params = params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = LinearModel(params['feature_type'], output_size=10).to(self.device)
        if params['test'] == True:
            self.model.load_state_dict(torch.load(params['model_path'], map_location=self.device))

        else:
            self.loss_fn = CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
            self.train_dataset = m_dataset(params['data_path'], params['feature_type'], dataType='train')
            self.train_dataloader = DataLoader(self.train_dataset, 
                                            batch_size=params['batch_size'],
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
            print('train data load finished!')
            print('train data number:{}'.format(len(self.train_dataset)))
            
            self.eval_dataset = m_dataset(params['data_path'], params['feature_type'], dataType='evaluate')
            self.eval_dataloader = DataLoader(self.eval_dataset, 
                                            batch_size=params['batch_size'],
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
            print('eval data load finished!')
            print('eval data number:{}'.format(len(self.eval_dataset)))

            self.test_dataset = m_dataset(params['data_path'], params['feature_type'], dataType='test')
            self.test_dataloader = DataLoader(self.test_dataset, 
                                            batch_size=params['batch_size'],
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
            print('test data load finished!')
            print('test data number:{}'.format(len(self.test_dataset)))

            self.logger = SummaryWriter(params['logger_dir'])
    
    def compute(self, data):
        return self.model(data)

    def train(self):
        for i in range(self.params['epoch']):
            self.model.train()
            print('epoch {}/{} start'.format(i+1, self.params['epoch']))
            for data, label in tqdm(self.train_dataloader):
                data = data.to(self.device)
                label = label.to(self.device)
                result = self.compute(data)
                loss = self.loss_fn(result, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.logger.add_scalar('Train/loss', loss.item()/data.shape[0], i)

            if i%self.params['eval_interval'] == 0:
                avg_loss = 0
                self.model.eval()
                print('start eval')
                accurate = 0
                count = 0
                for data, label in tqdm(self.eval_dataloader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    result = self.compute(data)
                    loss = self.loss_fn(result, label)
                    avg_loss += loss.item()
                    accurate += int((result.argmax(dim=1)==label).sum())
                    count += data.shape[0]
                self.logger.add_scalar('Eval/loss', avg_loss/count, i)
                self.logger.add_scalar('Eval/accuracy', accurate/count, i)
                print('eval loss:{}'.format(avg_loss/count))
                print('eval acc:{}'.format(accurate/count))
            if i%self.params['save_interval'] == 0:
                self.save(i)

    def save(self, epoch):
        if not os.path.exists(self.params['saving_dir']):
            os.mkdir(self.params['saving_dir'])
        torch.save(self.model.state_dict(), os.path.join(self.params['saving_dir'], '{}.pth'.format(epoch)))


if __name__ == '__main__':
    params = {
        'test': False,
        'data_path': 'D:\Git\AcousticSceneClassification\data',
        'feature_type': 'mel',
        'lr': 0.001,
        'batch_size': 64,
        'logger_dir': 'log',
        'saving_dir': 'results',
        'eval_interval': 1,
        'save_interval': 50,
        'epoch': 2000
    }
    solver = Solver(params)
    solver.train()


        


