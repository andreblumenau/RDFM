import numpy

def fm_get_p(X, W):
    xa = numpy.array([X])
    VX =  xa.dot(W)
    VX_square = (xa*xa).dot(W*W)
    phi = 0.5*(VX*VX - VX_square).sum()

    return phi

def matthews_coefficient(perf_table):
    tp = perf_table[0][0] #true positive
    tn = perf_table[1][1] #true negative
    fp = perf_table[0][1] #false positive
    fn = perf_table[1][0] #false negative
    
    M = (tp*tn - (fp*fn))/(numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00000001)
    return M

def table(X,Y):
    w, h = 2, 2
    table_t = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(X)):        
        a = (0 if X[i] < 0.5 else 1)
        b = (0 if Y[i] < 0.5 else 1)
        table_t[a][b] = table_t[a][b] + 1
    print(table_t)
    return table_t
    
def table_adapted(X,Y):

    X = [int(numpy.round(element*5,0)) for element in X]#numpy.round((X*5),0)
    Y = Y*5
    Y = [int(item) for sublist in Y for item in sublist]

    X = numpy.array(X)
    X = numpy.clip(X,0,5)
    X = X.tolist()
    
    w, h = 6, 6
    table_t = numpy.tile(0,(6,6))

    for i in range(len(X)):
        index_x = X[i]
        index_y = Y[i]
        table_t[index_x][index_y] = table_t[index_x][index_y]+1

    return table_t
        
def evaluate(x, y, w):
    print('evaluation')
    print('min y', min(y))
    print('max y', max(y))
    p_y = []

    for i in range(x.shape[0]): 
        p_y.append(fm_get_p(x[i], w))

    perf = table_adapted(p_y, y)
    #rmse = numpy.power((numpy.array(p_y)-numpy.array(y))/len(y))
    print('RMSE: ',)
    print('Performance: \n', perf)
    print('Accuracy:',(perf.trace()/x.shape[0]))
    print('MATTHEWS Coefficient:',matthews_coefficient(perf))
