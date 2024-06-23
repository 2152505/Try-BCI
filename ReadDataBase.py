import mne
import matplotlib.pyplot as plt
import numpy as np
import os

#文件路径和文件名称
root=os.getcwd()
data_path=os.path.join(root,'database','BCICIV_2b','B0101T.gdf')

#事件的对应关系
eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769':"cueLeft", '770':"cueRight", '781':"feedback", '783':"cueUnknown",
            '1023': "rejected", '1077': 'horizonEyeMove', '1078': "verticalEyeMove", '1079':"eyeRotation",'1081':"eyeBlinks",'32766':"startRun"}
# 读取原始数据
rawDataGDF = mne.io.read_raw_gdf(data_path,preload=True,eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])
# 定义数据通道和通道类型
ch_types = ['eeg', 'eeg', 'eeg','eog','eog','eog']
ch_names = ['EEG:Cz', 'EEG:C3', 'EEG:C4','EOG:ch01', 'EOG:ch02', 'EOG:ch03']
# 创建数据的描述信息
info = mne.create_info(ch_names=ch_names, sfreq=rawDataGDF.info['sfreq'], ch_types=ch_types)
# 创建数据结构体
data = np.squeeze(np.array([rawDataGDF['EEG:Cz'][0], rawDataGDF['EEG:C3'][0], rawDataGDF['EEG:C4'][0], rawDataGDF['EOG:ch01'][0] , rawDataGDF['EOG:ch02'][0] ,rawDataGDF['EOG:ch03'][0]]))
# 创建RawArray类型的数据
rawData = mne.io.RawArray(data, info)
rawData.plot()
# 获取事件
event,_= mne.events_from_annotations(rawDataGDF)
print(event)
event_id = {}
for i in _:#整理event_id
    event_id[eventDescription[i]] = _[i]
# 提取epoch
epochs = mne.Epochs(rawData,event,event_id, tmax=1.5 ,event_repeated = 'merge')
epochs.plot()
plt.show(block=True)  # 阻塞程序直到图形被关闭
ev_left = epochs['cueLeft']
ev_left.plot()
plt.show(block=True)  # 阻塞程序直到图形被关闭