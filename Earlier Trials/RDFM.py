import numpy
from numpy import genfromtxt
from numpy import arange
import time
import os
#import sys
#import winsound
import scipy
from scipy import special
    
def train(pTrainX, pTrainY, iterations, alpha, regularization, factors,w):

    alpha = numpy.array([alpha])    
    w = sgd_subset(pTrainX, pTrainY,iterations, alpha, regularization,w)

    return w

def sgd_subset(train_X, train_Y, iterations, alpha, regularization,weight_matrix):
    N = train_X.shape[0]#N = 6928 & 6928/866=8
    M = weight_matrix.shape[1]
    
    tensor_of_x_features = numpy.tile(0.0,(N,1,trainX.shape[1]))
    tensor_of_x_squared = numpy.tile(0.0,(N,trainX.shape[1],trainX.shape[1]))

    matrix_set_diag_to_zero = numpy.tile(1.0,(trainX.shape[1],trainX.shape[1]))
    numpy.fill_diagonal(matrix_set_diag_to_zero,0.0)
    for i in range(N):
        tensor_of_x_features[i]=train_X[i]
        tensor_of_x_squared[i]=train_X[i].dot(train_X[i])

    historical_gradient=numpy.tile(0.0,(weight_matrix.shape))
    tensor_of_x_squared = tensor_of_x_squared*matrix_set_diag_to_zero
    tensor_of_x_features_squared = tensor_of_x_features*tensor_of_x_features
    
    tensor_of_proto_vx = numpy.tile(0.0,(N,1,M))
    tensor_of_proto_square = numpy.tile(0.0,(N,1,M))
    vector_of_prediction = numpy.tile(0.0,N)
    vector_of_sum = numpy.tile(1.0,(M,1))
    vector_of_gradient = numpy.tile(0.0,N)
    
    weight_matrix_square = numpy.tile(0.0,(weight_matrix.shape))
    update_step = numpy.tile(0.0,(weight_matrix.shape))

    splits = 9
    taker = numpy.floor(N/splits).astype(numpy.int32)
    seed = 0
    
    idxs = numpy.linspace(0,taker,taker,dtype=numpy.int32)
    
    for i in range(iterations):
        seed = seed + 1
        numpy.random.seed(seed)
        random_idx_list = numpy.random.permutation(N)
        
        #skiper = 0        
        idxs = 0
        init = 0
        ending = 0
        
        for j in range(splits):
            init = j*taker
            ending = (j+1)*taker

            idxs = random_idx_list[init:ending]
        
            weight_matrix[numpy.abs(weight_matrix)<0.0000001]=0 
            weight_matrix_square = weight_matrix*weight_matrix
            tensor_of_proto_vx = numpy.tensordot(tensor_of_x_features[idxs],weight_matrix,axes=1)
            tensor_of_proto_square = numpy.tensordot(tensor_of_x_features_squared[idxs],weight_matrix_square,axes=1)
            vector_of_prediction = numpy.tensordot(((tensor_of_proto_vx*tensor_of_proto_vx) - tensor_of_proto_square),vector_of_sum,axes=1).sum(axis=1)*0.5
            b = train_Y[idxs]-vector_of_prediction
            #print(numpy.abs(b.mean()))
            vector_of_gradient = -2*b
            vrau = numpy.tensordot(tensor_of_x_squared[idxs],weight_matrix,axes=1)
            update_step = ((vector_of_gradient.T*vrau.T).T).sum(axis=0)+weight_matrix_square*regularization
    
            #ADAGRAD UPDATE
            historical_gradient += update_step * update_step
            weight_matrix -= alpha/(numpy.sqrt(historical_gradient)) * update_step#+0.000001
        
    return weight_matrix

def fm_gradient_sgd_trick(x_features, y_target, weights, regularization,meio,proto_x_matrix,proto_vx,proto_vx_square,proto_prediction):
    proto_x_matrix = x_features.T.dot(x_features)#(333,333)
    proto_x_matrix.setdiag(0)
    proto_vx =  x_features.dot(weights) #(1,4)    
    proto_vx_square = (x_features.multiply(x_features)).dot(weights.multiply(weights)) #(1,4)
    proto_prediction = meio.multiply(proto_vx.multiply(proto_vx) - proto_vx_square).sum()
    gradient = numpy.tanh(y_target-proto_prediction)        
    weights = weights.multiply(regularization) + (proto_x_matrix.dot(weights)).multiply(gradient)#+ (proto_x_matrix.dot(weights)).multiply(gradient)
    return  weights #(333,4)

def delete_column(array, *args):
    filtered_names = [x for x in array.dtype.names if x not in args]
    return array[filtered_names]

def softmax(X,W):
    xa = numpy.array([X])
    VX =  xa.dot(W)
    VX_square = (xa*xa).dot(W*W)
    phi = 0.5*(VX*VX - VX_square).sum()
    
    z = [0.0, 1.0]
    softmax_result = scipy.special.expit(phi)/numpy.sum(scipy.special.expit(z))
    if softmax_result > 0.5:
        return 1
    else:
        return 0
        
    
def fm_get_p(X, W):
    #print(W)
    xa = numpy.array([X])
    VX =  xa.dot(W)
    VX_square = (xa*xa).dot(W*W)
    phi = 0.5*(VX*VX - VX_square).sum()

    if phi > 0.5:
        return 1
    else:
        return 0

def table(X,Y):
    w, h = 2, 2
    table_t = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(X)):        
        a = (0 if X[i] < 0.5 else 1)
        b = (0 if Y[i] < 0.5 else 1)
        table_t[a][b] = table_t[a][b] + 1
    print(table_t)
    return table_t

def evaluate(x, y, w):
    print('evaluation')
    print('min y', min(y))
    print('max y', max(y))
    p_y = []

    for i in range(x.shape[0]): 
        p_y.append(softmax(x[i], w))#,meio))
    perf = table(p_y, y)
    print('Performance: ', perf)
    print('Accuracy:',(perf[0][0]+perf[1][1])/x.shape[0])
    print('MATTHEWS Coefficient:',MatthewsCoefficient(perf))

def MatthewsCoefficient(perf_table):
    tp = perf_table[0][0] #true positive
    tn = perf_table[1][1] #true negative
    fp = perf_table[0][1] #false positive
    fn = perf_table[1][0] #false negative
    
    M = (tp*tn - (fp*fn))/numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return M
    
aprendizado_fm = genfromtxt('/floyd/input/aprendizado/aprendizado_fm.csv', delimiter='\t', names=True)
teste_fm = genfromtxt('/floyd/input/teste/teste_fm.csv', delimiter='\t', names=True)

numpy.seterr(invalid='raise')
numpy.seterr(over='raise')
numpy.seterr(under='raise')
numpy.seterr(divide='raise')

trainX = aprendizado_fm.copy()
trainX = delete_column(trainX, "Vendido")
trainX = delete_column(trainX, "Agendado")
trainY = aprendizado_fm["Vendido"].copy()

validationX = teste_fm.copy()
validationX = delete_column(validationX, "Vendido")
validationX = delete_column(validationX, "Agendado")
validationY = teste_fm["Vendido"].copy()

trainY = trainY.view(numpy.float64).reshape(trainY.size,1)

trainX =  [[int(y) for y in x] for x in trainX]
trainX = numpy.clip(trainX,0, 1)

validationY =  validationY.view(numpy.float64).reshape(validationY.size,1)

validationX = [[int(y) for y in x] for x in validationX]
validationX = numpy.clip(validationX, 0, 1)

trainY = numpy.clip(trainY, 0, 1)
validationY = numpy.clip(validationY, 0, 1)

a_factors = 4
modelo =  numpy.random.ranf((trainX.shape[1], a_factors))
modelo = modelo / numpy.sqrt((modelo*modelo).sum())

skip = 0
end = 0
sp_split = 80
take = numpy.floor(trainX.shape[0]/sp_split).astype(numpy.int32)
start = time.time()
for i in range(sp_split):    
    skip = i*take    
    end = ((i+1)*take)      
    modelo = train( trainX[skip:end], trainY[skip:end], iterations=40, alpha=1/(100), regularization=1/(1000), factors=a_factors,w=modelo)

end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)

