import numpy
import scipy
from scipy import sparse

def fm_get_p(X, W):
    VX = X.dot(W)
    X_Squared = X.multiply(X)#scipy.sparse.csr_matrix.transpose(X)
    W_Squared = W*W
    VX_square = X_Squared.dot(W_Squared)
    
    phi = 0.5*(VX*VX-VX_square).sum()
    #print("phi.shape",phi.shape)
    #print("phi",phi)
    #tensor_of_x_squared.append(csr_matrix.transpose(training_features[i]).dot(training_features[i]))
    

    return phi

def matthews_coefficient(perf_table):
    tp = perf_table[0][0] #true positive
    tn = perf_table[1][1] #true negative
    fp = perf_table[0][1] #false positive
    fn = perf_table[1][0] #false negative
    
    M = (tp*tn - (fp*fn))/(numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00000001)
    return M
    
def table_ratings(X,Y):

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

def RMSE(p_y,y):
    y = y.todense()
    subtraction = numpy.subtract(numpy.array(p_y),numpy.transpose(y))
    rmse = numpy.abs(subtraction).sum()/len(y)
    
    print("rmse = ",rmse)
    return rmse

def error_by_index(p_y,y):
    y = numpy.array(y.todense())
    subtraction = numpy.subtract(numpy.array(p_y),numpy.transpose(y))
    print("subtraction.shape",subtraction.shape)
    enumerate = numpy.arange(0,subtraction.shape[1],1)
    error_by_index = numpy.vstack((numpy.abs(subtraction[0]),enumerate)).transpose()
    error_by_index = error_by_index[error_by_index[:,0].argsort()]
    
    print("error_by_index.shape",error_by_index.shape)
    return error_by_index
    
def evaluate(x, y, w):
    print('evaluation')
    print('min y', min(y))
    print('max y', max(y))
    p_y = []

    for i in range(x.shape[0]): 
        p_y.append(fm_get_p(x[i], w))
        
    #perf = table_ratings(p_y, y)
    

    #return RMSE, ACC, ConfusionMatrix
    rmse = RMSE(p_y,y)#,(perf.trace()/x.shape[0]),table_ratings(p_y, y)
    error_list = error_by_index(p_y,y)
    #print('RMSE: ',rmse)
    print('{"metric": "RMSE", "value": '+str(numpy.round(rmse,5))+'}')
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
    print('{"metric": "RMSE", "value": '+str(numpy.round(rmse,5))+'}')
    return rmse   

