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
import csv
import gc
   
path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)

a_factors = 5


#take - Defines the number of tensors that will be kept on memory simultaneously
#memory_split - How many parts the will be, depends on "take"
take = 2
memory_split =  numpy.floor(trainX.shape[0]/take).astype(numpy.int32)

start = time.time()

modelo =  numpy.random.ranf((trainX.shape[1], a_factors))
modelo = modelo / numpy.sqrt((modelo*modelo).sum())


iteration_error = 0 
last_iteration_error = 0
sample_error_limit = 0.0000001
sample_patience = 0
sample_patience_limit = 80

csv_delimiter = '	'
my_file = Path("/path/to/file")
f = open('errorPerIteration.csv','w',newline='')
writer = csv.writer(f, delimiter=csv_delimiter)
writer.writerow(["SAMPLE","ITERATION","RMSE"])
f.close()
    
skip = 0
end = 0   
    
#for j in range(memory_split):    
for j in range(2):
    skip = j*take    
    end = ((j+1)*take)      
    modelo,iteration_error,error_iter_array = learning( 
        trainX[skip:end], 
        trainY[skip:end], 
        iterations=20, alpha=1/(100),
        regularization=1/(1000),
        weight_matrix=modelo,
        patience_limit=10,
        iteration_error_diff_limit=0.0000001,
        batch_size=2)

    if numpy.abs(numpy.abs(iteration_error)-last_iteration_error) < sample_error_limit:
        sample_patience = sample_patience+1
    else:
        sample_patience = 0
        
    if sample_patience == sample_patience_limit:
        break;
    
    last_iteration_error = numpy.abs(iteration_error)
    
    f = open('errorPerIteration.csv','a',newline='')
    writer = csv.writer(f, delimiter=csv_delimiter)    
    
    for k in range(len(error_iter_array)):
        #f.write([j,k,error_iter_array[k],"\n"]) #Give your csv text here.
        writer.writerow([j,k,error_iter_array[k]])
    f.close()    
    gc.collect()
    
end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
