import pickle

with open('/home/dww/OD/weak_stream5/region_proposals.pkl', 'rb') as f:  # 以二进制读取模式打开文件
    data = pickle.load(f)
import numpy as np

# 假设 data 是加载的列表或元组
print("Number of arrays:", len(data))
#Number of arrays: 3
# ['boxes', 'indexes', 'scores']

print("Shape of first array:", data['indexes'])
print("Shape of second array:", data['boxes'])
print("Shape of third array:", data['scores'])
keys = list(data.keys())
print(keys)