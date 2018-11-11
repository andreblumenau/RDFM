import numpy
from numpy import genfromtxt
from numpy import arange
import time
import os
import winsound
import pre_process
from pre_process import process_csv_data
from metrics import matthews_coefficient
from metrics import table_adapted
from metrics import evaluate
import scipy
from scipy import special
import gc
#from sklearn import preprocessing
    # patience = 0
    # patience_limit = 5
    
def learning(train_X, train_Y, iterations, alpha, regularization,weight_matrix,patience_limit):
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

    splits = 1 #Split for performance
    taker = numpy.floor(N/splits).astype(numpy.int32)
    seed = 0
    
    idxs = numpy.linspace(0,taker,taker,dtype=numpy.int32)  

    patience = 0
    last_iteration_error = 0

    error_iter_array = numpy.tile(1,(iterations,1))     

    print("iterations = ",iterations)
    for i in range(iterations):
        print("iter = ",i)
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
            
            print(b.mean())
   
            error_sum = error_sum+b.mean()
            
            vector_of_gradient = -2*b
            vrau = numpy.tensordot(tensor_of_x_squared[idxs],weight_matrix,axes=1)
            update_step = ((vector_of_gradient.T*vrau.T).T).sum(axis=0)+weight_matrix_square*regularization
    
            #ADAGRAD UPDATE
            historical_gradient += update_step * update_step
            weight_matrix -= alpha/(numpy.sqrt(historical_gradient)) * update_step#+0.000001            

        error_iter_array[i] = error_sum/splits

        if numpy.abs(numpy.abs(error_iter_array[i]) - last_iteration_error) < 0.0000001:
          patience = patience+1
        else:
          patience = 0        
          
        if patience == patience_limit:
          break
        
        last_iteration_error = numpy.abs(error_iter_array[i])
        
    return weight_matrix,error_iter_array.mean()
   
path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)

a_factors = 5

skip = 0
end = 0
sp_split = 500 #Split for Memory
take = numpy.floor(trainX.shape[0]/sp_split).astype(numpy.int32)
start = time.time()

modelo =  numpy.random.ranf((trainX.shape[1], a_factors))
modelo = modelo / numpy.sqrt((modelo*modelo).sum())

sp_error = 0
sp_patience = 0
sp_patience_limit = 80
sp_last_iteration_error = 0



#for i in range(sp_split):    
for i in range(10):
    skip = i*take    
    end = ((i+1)*take)      
    modelo,sp_error = learning( 
        trainX[skip:end], 
        trainY[skip:end], 
        iterations=20, alpha=1/(100),
        regularization=1/(1000),
        weight_matrix=modelo,
        patience_limit=10)

    if numpy.abs(numpy.abs(sp_error)-sp_last_iteration_error) < 0.0000001:
        sp_patience = sp_patience+1
    else:
        sp_patience = 0
        
    if sp_patience == sp_patience_limit:
        break;
    
    sp_last_iteration_error = numpy.abs(sp_error)

    gc.collect()
    
end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
