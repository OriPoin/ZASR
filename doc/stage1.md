# 阶段一

## 音频数据预处理

以 aidatatang_200zh 的 dev 部分为例

使用的所有数据集均不含噪声，输入为背景噪声较小的普通话音频，并将不同数据集的音频统一格式化为单通道音频。可以测试不同精度下的正确率与速度的变化，格式化为：采样率 8kHz/16kHz，精度： 16bit/8bit 整形。大部分数据集都是 16kHz 16bit 整形的 wav 音频文件。

### 统一格式

这里格式化为 16kHz 16bit 整形（也是大部分数据集默认格式）

```python
import numpy as np
from zasr import utils
import matplotlib.pyplot as plt

DataPath = "/data"
DataSetName = "aidatatang_200zh"
DevDataDir = "corpus/dev/G0002"
WavName = "T0055G0002S0001"

WavDir = DataPath+"/"+DataSetName+"/"+DevDataDir+"/"+WavName+".wav"
DataSR = 16000
DataPrecision = np.int16
RowData = utils.WavFormat(WavDir, DataSR, DataPrecision)
time = np.linspace(0, RowData.size/DataSR, RowData.size)

plt.plot(time, RowData)
plt.ylim(np.iinfo(DataPrecision).min,np.iinfo(DataPrecision).max)
plt.show()
```

### 预加重

为了减轻声音录制与声音传播时的高频衰减，使用一个高通滤波器对声音信号预加重。

$$H(z)=1 - a\times z^{-1}$$

转换到时域的离散表达式为：

$$y(t)=x(t)-a\times x(t-1)$$

这里 $a$ 取 0.97

$$a=0.97$$

```python
pre_emphasis=0.97
PreEmpData=utils.PreEmphasis(RowData,pre_emphasis)
time = np.linspace(0, PreEmpData.size/DataSR, PreEmpData.size)
plt.plot(time, PreEmpData)
plt.ylim(np.iinfo(DataPrecision).min,np.iinfo(DataPrecision).max)
plt.show()
```

### 分帧

16kHz 采样率下，窗口长度 25ms,400 个样点。步长 10ms,160 个样点。窗口数向上取整获得总样点，先补零，然后分帧。

```python
print("pading with zero")
FrameSize = 25  # time ms
FrameSize = int(DataSR*(FrameSize/1000))  # sample point
FrameStep = 10
FrameStep = int(DataSR*(FrameStep/1000))
Frames = utils.GenFrames(PreEmpData,FrameSize,FrameStep,DataSR)
print(Frames)
print("first frame")
time = np.linspace(0, Frames[0].size/DataSR, Frames[0].size)
plt.plot(time, Frames[0])
plt.show()
```

### 加窗

由于分帧相当于对信号加矩形窗然后截取，会有能量泄漏，使用汉明窗对每一帧加强中心的信号，减弱边界的能量减弱。

```python
Frames = utils.HanmmingWindow(Frames,FrameSize)

print("fisrt frame")
plt.plot(Frames[0])
plt.show()
```

### FFT

使用快速傅立叶变换，获取单边幅度谱

```python
FramePower = utils.GenSpectrum(Frames,FrameSize)
plt.plot(FramePower[0])
plt.show()
```

### Mel 滤波

使用 Mel 滤波组滤波，滤波器数量通常取 22-40，26 是标准，这里取 40。

滤波器数量过少精度降低，数量过多，归一化后梯度过低

40 组滤波器如下图所示

```python
nfilt=40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (DataSR / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((FrameSize + 1) * hz_points / DataSR)
fbank = np.zeros((nfilt, int(FrameSize/2)))
for m in range(1,nfilt+1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
plt.plot(fbank.T)
plt.show()
```

使用该滤波器过滤后得到

```python
nfilt=40
filter_banks = utils.MelFilter(FramePower,FrameSize,DataSR,nfilt)
plt.imshow(filter_banks.T,cmap='turbo',origin='lower')
plt.show()
```

对能量部分补零取 dB，获得 logfbank。

为了平衡频谱并改善信噪比（SNR），我们可以简单地从所有帧中减去每个系数的平均值。

```python
logfbank = utils.GenLogFBank(filter_banks)
plt.imshow(logfbank.T,cmap='turbo',origin='lower')
plt.show()
```

## 文本数据预处理



## 模型的训练

模型分为两个部分，一个是声音特征的编码器 encoder,一个是预测网络 prediction network，预测网络和联合网络 joint network 作为解码器输出预测的汉字。

### 编码器 encoder



### 预测网络 prediction network

### 联合网络 joint network

## 模型的验证

## 分析与瓶颈
