import numpy
from numpy import genfromtxt
import time
import os
import winsound
import pre_process
import cpu_learning
from pathlib import Path
from pre_process import process_csv_data
from metrics import matthews_coefficient
from metrics import table_adapted
from metrics import evaluate
from cpu_learning import learning
import gc
   
path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)

a_factors = 5
skip = 0
end = 0
memory_split = 500 #Split for Memory
take = numpy.floor(trainX.shape[0]/memory_split).astype(numpy.int32)
start = time.time()

modelo =  numpy.random.ranf((trainX.shape[1], a_factors))
modelo = modelo / numpy.sqrt((modelo*modelo).sum())

iteration_error = 0
sample_patience = 0
sample_patience_limit = 80
last_iteration_error = 0
sample_error_limit = 0.0000001

my_file = Path("/path/to/file")
if my_file.is_file() == false:
    f = open('errorPerIteration.csv','w')
    f.write("")
    f.close()
    

#for j in range(memory_split):    
for j in range(10):
    skip = j*take    
    end = ((j+1)*take)      
    modelo,iteration_error,error_iter_array = learning( 
        trainX[skip:end], 
        trainY[skip:end], 
        iterations=20, alpha=1/(100),
        regularization=1/(1000),
        weight_matrix=modelo,
        patience_limit=10,
        error_diff_limit=0.0000001,
        performance_splits=1)

    if numpy.abs(numpy.abs(iteration_error)-last_iteration_error) < sample_error_limit:
        sample_patience = sample_patience+1
    else:
        sample_patience = 0
        
    if sample_patience == sample_patience_limit:
        break;
    
    last_iteration_error = numpy.abs(iteration_error)
    
    f = open('errorPerIteration.csv','a')
    for k in range(len(error_iter_array)):
        print(error_iter_array[k])
        f.write(str(error_iter_array[k])+",\n") #Give your csv text here.
    f.close()    
    gc.collect()
    
end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
