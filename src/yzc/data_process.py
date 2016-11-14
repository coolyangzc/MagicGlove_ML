import numpy as np

def load_data(filename, start = 0, end = -1, 
              shuffle = True, coordinate = 'relative', validation_split = 0.1):   

    #Loading
    fd = file(filename)
    data = fd.readlines()
    X = []
    Y = [] 
    if (end == 0):
        end = len(data)
    for line in data[start:end]:
        line_data = map(float, line.split(' '))
        a = np.array(line_data[1:2*37*65+1])
        X.append(a.reshape((2, 37, 65), order='F'))
        Y.append(line_data[2*37*65+1:])
    X = np.array(X)
    Y = np.array(Y)
    
    #Processing
    if (coordinate == 'relative'):
        for label in Y:
            for i in range(3, len(label)):
                label[i] -= label[i%3]
        Y = np.delete(Y, np.s_[:3:], axis = 1)
    else:
        coordinate = 'absolute'
    if (shuffle):
        r = np.random.permutation(len(Y))
        X = X[r]
        Y = Y[r]
    sp = (1 - validation_split) * len(Y)
    X_train, X_test = X[:sp], X[sp:]
    Y_train, Y_test = Y[:sp], Y[sp:]
    
    #Output
    print ("successful loaded %d[%d:%d] datas from %s"%(len(Y_train), start, len(data) + end if end < 0 else end, filename))
    print ('shuffle = %r'%(shuffle))
    print ('coordinate = %s'%(coordinate))
    print ('validation_split = %f'%(validation_split))
    print 'X_train.shape =', X_train.shape
    print 'Y_train.shape =', Y_train.shape
    print 'X_test.shape =', X_test.shape
    print 'Y_test.shape =', Y_test.shape
    return (X_train, Y_train), (X_test, Y_test)