import numpy
from numpy import genfromtxt
from numpy import arange
import time
import os
import winsound
import pre_process
from pre_process import process_csv_data
#from sklearn import preprocessing

    
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

    splits = 9 #Split for performance
    taker = numpy.floor(N/splits).astype(numpy.int32)
    seed = 0
    
    idxs = numpy.linspace(0,taker,taker,dtype=numpy.int32)  

    patience = 0
    patience_limit = 5
    last_iteration_error = 0

    error_iter_array = numpy.tile(1,(iterations,1))     

    for i in range(iterations):
        seed = seed + 1
        numpy.random.seed(seed)
        random_idx_list = numpy.random.permutation(N)

        idxs = 0
        init = 0
        ending = 0
        error_sum = 0
        
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
   
            error_sum = error_sum+b.mean()
            
            vector_of_gradient = -2*b
            vrau = numpy.tensordot(tensor_of_x_squared[idxs],weight_matrix,axes=1)
            update_step = ((vector_of_gradient.T*vrau.T).T).sum(axis=0)+weight_matrix_square*regularization
    
            #ADAGRAD UPDATE
            historical_gradient += update_step * update_step
            weight_matrix -= alpha/(numpy.sqrt(historical_gradient)) * update_step#+0.000001            

        #print(error_sum)
        #print(splits)
        error_iter_array[i] = error_sum/splits

        #error_iter_array[i] = error_sum/splits

        if numpy.abs(numpy.abs(error_iter_array[i]) - last_iteration_error) < 0.0000001:
            patience = patience+1
        else:
            patience = 0

        if patience == patience_limit:
            break

        last_iteration_error = numpy.abs(error_iter_array[i])
        
    return weight_matrix,error_iter_array.mean()

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

def softmax(X,W,Y_VECTOR_EXPIT_SUM):
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
    
    y_expit=numpy.sum(scipy.special.expit(y))

    for i in range(x.shape[0]): 
        p_y.append(softmax(x[i], w,y_expit))#,meio))
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

# big_file = open("C:/PosGrad/Experimentos/aprendizado_fm.csv",'r')
# lines = big_file.readlines()[:10]
# big_file.close()

 # eita = genfromtxt(lines, delimiter='\t', names=True)
    
# aprendizado_fm = genfromtxt('C:\PosGrad\Movielens1M\data_processed_ 1 .csv', delimiter=',', names=True)
# teste_fm = genfromtxt('C:\PosGrad\Movielens1M\data_processed_ 1 .csv', delimiter=',', names=True)

# numpy.seterr(invalid='raise')
# numpy.seterr(over='raise')
# numpy.seterr(under='raise')
# numpy.seterr(divide='raise')

# trainX = aprendizado_fm.copy()
# trainX = delete_column(trainX, "Vendido")
# #trainX = delete_column(trainX, "Agendado")
# trainY = aprendizado_fm["Rating"].copy()

# validationX = teste_fm.copy()
# validationX = delete_column(validationX, "Rating")
# #validationX = delete_column(validationX, "Agendado")
# validationY = teste_fm["Rating"].copy()

# trainY = trainY.view(numpy.float64).reshape(trainY.size,1)

# trainX =  [[int(y) for y in x] for x in trainX]
# trainX = numpy.clip(trainX,0, 1)

# validationY =  validationY.view(numpy.float64).reshape(validationY.size,1)

# validationX = [[int(y) for y in x] for x in validationX]
# validationX = numpy.clip(validationX, 0, 1)

# training_bias_vector = numpy.tile(1,(trainX.shape[0],1))
# validation_bias_vector = numpy.tile(1,(validationX.shape[0],1))

# trainX = numpy.hstack((training_bias_vector,trainX))
# validationX = numpy.hstack((validation_bias_vector,validationX))

# trainX = preprocessing.scale(trainX)
# validationX = preprocessing.scale(validationX)

# trainY = numpy.clip(trainY, 0, 1)
# validationY = numpy.clip(validationY, 0, 1)


path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)

a_factors = 4

skip = 0
end = 0
sp_split = 240 #Split for Memory
take = numpy.floor(trainX.shape[0]/sp_split).astype(numpy.int32)
start = time.time()

modelo =  numpy.random.ranf((trainX.shape[1], a_factors))
modelo = modelo / numpy.sqrt((modelo*modelo).sum())

sp_error = 0
sp_patience = 0
sp_patience_limit = 80
sp_last_iteration_error = 0

for i in range(sp_split):    
    skip = i*take    
    end = ((i+1)*take)      
    modelo,sp_error = sgd_subset( trainX[skip:end], trainY[skip:end], iterations=1, alpha=1/(100), regularization=1/(1000), weight_matrix=modelo)

    if numpy.abs(numpy.abs(sp_error)-sp_last_iteration_error) < 0.0000001:
        sp_patience = sp_patience+1
    else:
        sp_patience = 0
        
    if sp_patience == sp_patience_limit:
        break;
    
    sp_last_iteration_error = numpy.abs(sp_error)

end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
