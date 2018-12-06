import cupy
import numpy

def fm_get_p(X, W):
    xa = cupy.array([X])
    VX =  xa.dot(W)
    VX_square = (xa*xa).dot(W*W)
    phi = 0.5*(VX*VX - VX_square).sum()

    return phi

def matthews_coefficient(perf_table):
    tp = perf_table[0][0] #true positive
    tn = perf_table[1][1] #true negative
    fp = perf_table[0][1] #false positive
    fn = perf_table[1][0] #false negative
    
    M = (tp*tn - (fp*fn))/(cupy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00000001)
    return M
    
def table_ratings(X,Y):

    X = [int(cupy.round(element*5,0)) for element in X]#cupy.round((X*5),0)
    Y = Y*5
    Y = [int(item) for sublist in Y for item in sublist]

    X = cupy.array(X)
    X = cupy.clip(X,0,5)
    X = X.tolist()
    
    w, h = 6, 6
    table_t = cupy.tile(0,(6,6))

    for i in range(len(X)):
        index_x = X[i]
        index_y = Y[i]
        table_t[index_x][index_y] = table_t[index_x][index_y]+1

    return table_t

def RMSE(p_y,y):
    subtraction = cupy.subtract(cupy.array(p_y),cupy.transpose(y))
    rmse = cupy.abs(subtraction).sum()/len(y)
    return rmse

def error_by_index(p_y,y):
    #p_y = numpy.array(p_y)
    subtraction = cupy.subtract(cupy.array(p_y),cupy.transpose(y))
    enumerate = cupy.arange(0,subtraction.shape[1],1)
    error_by_index = cupy.vstack((cupy.abs(subtraction[0]),enumerate)).transpose()
    error_by_index = error_by_index[error_by_index[:,0].sort()]
    return error_by_index
    
def evaluate(x, y, w):
    print('evaluation')
    print('min y', min(y))
    print('max y', max(y))
    p_y = []

    for i in range(x.shape[0]): 
        p_y.append(float(fm_get_p(x[i], w)))
        
    #perf = table_ratings(p_y, y)
    

    #return RMSE, ACC, ConfusionMatrix
    p_y = cupy.array(p_y)
    y = cupy.array(y)
    
    rmse = RMSE(p_y,y)#,(perf.trace()/x.shape[0]),table_ratings(p_y, y)[
    print("rmse = ", rmse)
    print("rmse shape = ",rmse.shape)
    error_list = error_by_index(p_y,y)
    #print('RMSE: ',rmse)
    print('{"metric": "RMSE", "value": '+str(round(float(rmse),5))+'}')
    return rmse,error_list
    #print('Performance: \n', perf)
    #print('Accuracy:',(perf.trace()/x.shape[0]))
    #print('MATTHEWS Coefficient:',matthews_coefficient(perf))
    
def evaluate_rmse(x, y, w):
    p_y = []

    for i in range(x.shape[0]): 
        p_y.append(fm_get_p(x[i], w))

    rmse = RMSE(p_y,y)#,(perf.trace()/x.shape[0]),table_ratings(p_y, y)
    # print("x = \n",x)
    # print("y = \n",y)
    #print('RMSE: ',rmse)
    print('{"metric": "RMSE", "value": '+str(cupy.round(rmse,5))+'}')
    return rmse   

