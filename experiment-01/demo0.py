from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
fs, dataAll = wavfile.read('1.wav') # 读取 wav 音频文件
data = dataAll.T[0] # 获取第一个音频通道的数据(exp1 的文件只有一个通道)
plt.plot(data)
plt.show()