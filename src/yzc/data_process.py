import numpy as np

def load_data(filename, start = 0, end = -1, mission = 'magicglove',
              shuffle = True, select = 128, coordinate = 'relative', validation_split = 0.1):   

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
        a = a.reshape(2, 37*65, order = 'F')
        a = a.reshape(2, 37, 65)
        cnt = 0
        flag = False
        for i in range(37):
            for j in range(65):
                if (a[0,i,j] != 0):
                    cnt += 1
                    if (cnt >= select):
                        flag = True
                        break
        if (flag):    
            X.append(a)
            Y.append(line_data[2*37*65+1:])
    X = np.array(X)
    Y = np.array(Y)
    
    #Processing
    if (mission == 'magicglove'):
        if (coordinate == 'relative'):
            for label in Y:
                for i in range(3, len(label)):
                    label[i] -= label[i%3]
            Y = np.delete(Y, np.s_[:3:], axis = 1)
        else:
            coordinate = 'absolute'
    if (shuffle):
        for i in range(len(Y)):
            j = np.random.randint(len(Y))
            X[i], X[j] = X[j], X[i].copy()
            Y[i], Y[j] = Y[j], Y[i].copy()
        '''
        r = np.random.permutation(len(Y))
        X = X[r]
        Y = Y[r]
        '''
        
    sp = int((1 - validation_split) * len(Y))
    X_train, X_test = X[:sp], X[sp:]
    Y_train, Y_test = Y[:sp], Y[sp:]
    
    #Output
    print ("successful loaded %d[%d:%d] datas from %s"%(len(Y), start, len(data) + end if end < 0 else end, filename))
    print ('shuffle = %r'%(shuffle))
    print ('coordinate = %s'%(coordinate))
    print ('validation_split = %f'%(validation_split))
    print 'X_train.shape =', X_train.shape
    print 'Y_train.shape =', Y_train.shape
    print 'X_test.shape =', X_test.shape
    print 'Y_test.shape =', Y_test.shape
    return (X_train, Y_train), (X_test, Y_test)