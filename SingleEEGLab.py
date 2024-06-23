import numpy as np
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
import os
import logging
import matplotlib.pyplot as plt


def test(type=1):
    root = os.getcwd()
    data_path = os.path.join(root, 'database','sample_data','eeglab_data.set')

    """数据地址（需要改成你自己的数据地址，在EEGLAB文件夹的sample_data文件夹下）
    data_path = "/Users/zitonglu/Desktop/EEG/eeglab14_1_2b/sample_data/eeglab_data.set"
    MNE-Python中对多种格式的脑电数据都进行了支持：
    *** 如数据后缀为.set (来自EEGLAB的数据)
        使用mne.io.read_raw_eeglab()
    *** 如数据后缀为.vhdr (BrainVision系统)
        使用mne.io.read_raw_brainvision()
    *** 如数据后缀为.edf
        使用mne.io.read_raw_edf()
    *** 如数据后缀为.bdf (BioSemi放大器)
        使用mne.io.read_raw_bdf()
    *** 如数据后缀为.gdf
        使用mne.io.read_raw_gdf()
    *** 如数据后缀为.cnt (Neuroscan系统)
        使用mne.io.read_raw_cnt()
    *** 如数据后缀为.egi或.mff
        使用mne.io.read_raw_egi()
    *** 如数据后缀为.data
        使用mne.io.read_raw_nicolet()
    *** 如数据后缀为.nxe (Nexstim eXimia系统)
        使用mne.io.read_raw_eximia()
    *** 如数据后缀为.lay或.dat (Persyst系统)
        使用mne.io.read_raw_persyst()
    *** 如数据后缀为.eeg (Nihon Kohden系统)
        使用mne.io.read_raw_nihon()
    """
    # 读取数据
    raw = mne.io.read_raw_eeglab(data_path, preload=True)

    #查看原始数据信息
    print(raw)
    print(raw.info)

    # locs文件地址
    locs_info_path = os.path.join(root,'database','sample_data','eeglab_chan32.locs')
    # 读取电极位置信息
    montage = mne.channels.read_custom_montage(locs_info_path)
    # 读取正确的导联名称
    new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
    # 读取旧的导联名称
    old_chan_names = raw.info["ch_names"]
    # 创建字典，匹配新旧导联名称
    chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
    # 更新数据中的导联名称
    raw.rename_channels(chan_names_dict)
    # 传入数据的电极位置信息
    raw.set_montage(montage)
    print("===============================================================================================")
    print("locs文件地址")
    print(locs_info_path)
    locs_info_path = os.path.join(root,'database','sample_data','eeglab_chan32.locs')
    print("===============================================================================================")
    print("读取电极位置信息")
    print(montage)
    montage = mne.channels.read_custom_montage(locs_info_path)
    print("===============================================================================================")
    print("读取正确的导联名称")
    print(new_chan_names)
    new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
    print("===============================================================================================")
    print("读取旧的导联名称")
    print(old_chan_names)
    old_chan_names = raw.info["ch_names"]
    print("===============================================================================================")
    print("创建字典，匹配新旧导联名称")
    print(chan_names_dict)
    chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
    print("===============================================================================================")
    print("更新数据中的导联名称")
    print(chan_names_dict)
    raw.rename_channels(chan_names_dict)
    print("===============================================================================================")
    print("传入数据的电极位置信息")
    print(montage)
    raw.set_montage(montage)
    print("===============================================================================================")

    # MNE中一般默认将所有导联类型设成eeg
    # 将两个EOG导联的类型设定为eog
    chan_types_dict = {"EOG1":"eog", "EOG2":"eog"}
    raw.set_channel_types(chan_types_dict)

    # 打印修改后的数据相关信息
    print(raw.info)
    raw.plot(duration=5, n_channels=32, clipping=None)
    plt.show(block=True)  # 阻塞程序直到图形被关闭
    raw.plot_psd(average=True)
    plt.show(block=True)  # 阻塞程序直到图形被关闭
    raw.plot_sensors(ch_type='eeg', show_names=True)
    plt.show(block=True)  # 阻塞程序直到图形被关闭
    # raw.plot_psd_topo()
    # plt.show(block=True) 
    # 阻塞程序直到图形被关闭
    raw = raw.notch_filter(freqs=(60))
    raw.plot_psd(average=True)
    plt.show(block=True)  # 阻塞程序直到图形被关闭

    raw = raw.filter(l_freq=0.1, h_freq=30)
    raw = raw.filter(l_freq=0.1, h_freq=30, method='iir')
    raw.plot_psd(average=True)
    plt.show(block=True)  # 阻塞程序直到图形被关闭

    fig = raw.plot(duration=5, n_channels=32, clipping=None)  
    fig.canvas.key_press_event('a')
    plt.show(block=True)  # 阻塞程序直到图形被关闭

    ica = ICA(max_iter='auto')
    raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
    ica.fit(raw_for_ica)
    ica.plot_sources(raw_for_ica)

    ica.plot_components()
    plt.show(block=True)
    
    ica.plot_overlay(raw_for_ica, exclude=[1])
    plt.show(block=True)
    
    ica.plot_properties(raw, picks=[1, 16])
    plt.show(block=True)

    # 设定要剔除的成分序号
    ica.exclude = [1]
    # 应用到脑电数据上
    ica.apply(raw)
    
    raw.plot(duration=5, n_channels=32, clipping=None)
    plt.show(block=True)


if __name__ == '__main__':
    test()