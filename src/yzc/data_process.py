import numpy as np

def load_data(filename, start = 0, end = -1, 
              shuffle = True, coordinate = 'relative'):   

    #Loading
    fd = file(filename)
    data = fd.readlines()
    X_train = []
    Y_train = [] 
    if (end == 0):
        end = len(data)
    for line in data[start:end]:
        line_data = map(float, line.split(' '))
        a = np.array(line_data[1:2*37*65+1])
        X_train.append(a.reshape((2, 37, 65), order='F'))
        Y_train.append(line_data[2*37*65+1:])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    #Processing
    if (coordinate == 'relative'):
        for label in Y_train:
            for i in range(3, len(label)):
                label[i] -= label[i%3]
    else:
        coordinate = 'absolute'
    if (shuffle):
        r = np.random.permutation(len(Y_train))
        X_train = X_train[r]
        Y_train = Y_train[r]
    
    #Output
    print ("successful loaded %d[%d:%d] datas from %s"%(len(Y_train), start, len(data) + end if end < 0 else end, filename))
    print ('shuffle = %r'%(shuffle))
    print ('coordinate = %s'%(coordinate))
    return X_train, Y_train