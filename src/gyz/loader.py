import os
import numpy as np

INP_CHANNEL = 2
INP_R = 37
INP_C = 65
OUP_DIM = 60
NORMAL = 500

def load(file_name = 'data.txt', st = 0, en = -1):
	lines = file(file_name).readlines()
	data_size = en - st
	if st == -1:
		data_size = len(lines) - st
	inp = np.empty((data_size, INP_CHANNEL, INP_R, INP_C), dtype = 'int32')
	oup = np.empty((data_size, OUP_DIM), dtype = 'float')
	for i in range(st, en):
		values = map(float, lines[i].split(' '))
		for channel in range(INP_CHANNEL):
			for r in range(INP_R):
				for c in range(INP_C):
					inp[i - st, channel, r, c] = int(values[1 + r * INP_CHANNEL * INP_C + c * INP_CHANNEL + channel])
		for dim in range(OUP_DIM):
			oup[i - st, dim] = (float(values[1 + INP_CHANNEL * INP_R * INP_C + 3 + dim]) - float(values[1 + INP_CHANNEL * INP_R * INP_C + dim % 3])) / NORMAL
	return inp, oup
