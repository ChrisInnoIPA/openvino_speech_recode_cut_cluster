# from pydub import AudioSegment
# import wave

# #folder_name = "./sample_audio/"
# file_name = "tokyoleft.wav"
# sound = AudioSegment.from_wav(file_name)
# start_time = "0:00"
# stop_time = "0:42"
# print("time:",start_time,"~",stop_time)
# start_time = (int(start_time.split(':')[0])*60+int(start_time.split(':')[1]))*1000
# stop_time = (int(stop_time.split(':')[0])*60+int(stop_time.split(':')[1]))*1000
# print("ms:",start_time,"~",stop_time)
# word = sound[start_time:stop_time]
# save_name = "word"+file_name[6:]
# print(save_name)
# word.export(save_name, format="wav",tags={'artist': 'AppLeU0', 'album': save_name[:-4]})

#-------------------------------------------------------------#

# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import random
# import sys
# import wave

# #name = sys.argv[1]
# name = "output"
# file_name = name + ".wav"
# sound = AudioSegment.from_wav(file_name)
	 
# chunks = split_on_silence(sound,min_silence_len=700,silence_thresh=-50)#silence time:700ms and silence_dBFS<-70dBFS

# words = chunks[2:] #first and second are not words.
	 
# len1 = len(words)

# new = AudioSegment.from_wav(file_name)#.empty()
# silence = AudioSegment.silent(duration=1000)#1000ms silence

# order = range(len1)
# random.shuffle(order)
# print(order)
# comments = ""

# for i in order:
#     new += words[i]+silence
#     comments += str(i)+","

# save_name = file_name.split(".")[0]+"-random{0}.".format(random.randrange(0,9))+file_name.split(".")[1]
# new.export(save_name, format="wav",tags={'artist': 'AppLeU0', 'album': file_name, 'comments': comments[:-1]})

#-------------------------------------------------------------#

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def cutaudio():
    # 初始化
    audiopath = "output.wav"
    audiotype = 'wav' #如果wav、mp4其他格式参看pydub.AudioSegment的API
    # 读入音频
    print('Read audio')
    sound = AudioSegment.from_file(audiopath, format="wav")
    sound = sound[:3*60*1000] #如果文件较大，先取前3分钟测试，根据测试结果，调整参数
    # 分割 
    print('Cut audio')
    chunks = split_on_silence(sound,min_silence_len=300,silence_thresh=-50)#min_silence_len: 拆分语句时，静默满0.3秒则拆分。silence_thresh：小于-70dBFS以下的为静默。
    # 创建保存目录
    targetpath = '/opt/intel/openvino_2021.2.185/inference_engine/demos/python_demos/speech_recognition_demo/myrecodeaudio/'
    filepath = os.path.split(audiopath)[0]
    chunks_path = filepath + targetpath
    if not os.path.exists(chunks_path):os.mkdir(chunks_path)
    # 保存所有分段
    print('Save audio')
    for i in range(len(chunks)):
        new = chunks[i]
        save_name = chunks_path+'%04d.%s'%(i,audiotype)
        new.export(save_name, format=audiotype)
        print('%04d'%i,len(new))
    print('Save finish')