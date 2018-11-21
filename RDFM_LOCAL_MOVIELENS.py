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

    weight_matrices = []

    for j in range(number_of_instances):
        trainX,trainY,validationX,validationY,validation_indexes = data_handler.process_csv_data(lineStart = sample_start, lineEnd = sample_end)
    
        instance_list[j].learn(trainX,trainY)
        rmse = instance_list[j].predict(validationX,validationY)
        
        weight_matrices.append(instance_list[j].model)
    
        sample_start = sample_start + dataset_partition_size
        sample_end = sample_end + dataset_partition_size
        
    tardigrade_matrices = numpy.array(weight_matrices)    
    print("tardigrade_matrices.shape",tardigrade_matrices.shape)
        
    for j in range(number_of_instances):
        #numpy.delete creates a new list without the instance_list[j] model
        instance_list[j].tardigrade(data_handler,numpy.delete(tardigrade_matrices,j,axis=0))

end = time.time()    
            
print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
#evaluate(validationX,validationY,factorization_machine.model)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
