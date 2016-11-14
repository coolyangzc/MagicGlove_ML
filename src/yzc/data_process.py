import numpy as np

def load_data(filename, start = 0, end = -1):   
    fd = file(filename)
    data = fd.readlines()
    X_train = []
    Y_train = [] 
    if (end == 0):
        end = len(data)
    for line in data[start:end]:
        line_data = map(float, line.split(' '))
        a = np.array(line_data[1:37*65*2+1])
        X_train.append(a.reshape((2, 37, 65), order='F'))
        Y_train.append(line_data[37*65*2+1:])
    print ("Successful loaded %d datas from %s"%(len(Y_train), filename))
    return X_train, Y_train