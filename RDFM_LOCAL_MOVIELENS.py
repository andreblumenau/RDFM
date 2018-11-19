import numpy
import time
import winsound
import pre_process
import factorization_machine
from factorization_machine import FactorizationMachine
from pre_process import DataProcessing
from metrics import evaluate
import csv

number_of_instances = 2
instance_list = []


for i in range(number_of_instances):
    factorization_machine = FactorizationMachine(
        iterations                      = 10,
        learning_rate                   = 1/(100),
        latent_vectors                  = 2,
        regularization                  = 1/(1000),
        slice_size                      = 2,
        batch_size                      = 2,
        slice_patience                  = 5,
        iteration_patience              = 5,
        slice_patience_threshold        = 0.0000001,
        iteration_patience_threshold    = 0.0000001)
        
    instance_list.append(factorization_machine)
            
data_handler = DataProcessing(
    path = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv",
    delimiter_char = ",",
    target_column = "Rating")                
    
dataset_size = 999    
sample_start = 0
sample_end = 0
turns = 2
start = time.time()

dataset_partition_size = int(numpy.floor(999/(turns*number_of_instances)))
correction_offset = dataset_size - dataset_partition_size*turns*number_of_instances
sample_end = dataset_partition_size + correction_offset

for i in range(turns):
    for j in range(number_of_instances):
        trainX,trainY,validationX,validationY,validation_indexes = data_handler.process_csv_data(lineStart = sample_start, lineEnd = sample_end)
    
        instance_list[j].learn(trainX,trainY)
        rmse,error_by_index = evaluate(validationX,validationY,instance_list[j].model)
        print("5 índices com menores erros = ",error_by_index[0:5,1].astype(numpy.int32))
        print("5 índices com maiores erros = ",error_by_index[(len(error_by_index)-5):len(error_by_index),1].astype(numpy.int32))
    
        sample_start = sample_start + dataset_partition_size
        sample_end = sample_end + dataset_partition_size

#trainX,trainY,validationX,validationY = data_handler.process_csv_data(lineStart  = 0,lineEnd    = 999)
#factorization_machine.learn(trainX,trainY)
end = time.time()    
            
print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
#evaluate(validationX,validationY,factorization_machine.model)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
