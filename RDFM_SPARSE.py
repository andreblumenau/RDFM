import numpy
import time
import winsound
import pre_process_sparse
import factorization_machine_sparse
from factorization_machine_sparse import FactorizationMachine
from pre_process_sparse import DataProcessing
from metrics_sparse import evaluate
import csv
import random
from random import shuffle

dataset_size = 999    
sample_start = 0
sample_end = 0
turns = 2
number_of_instances = 6
number_of_random_failed = 1
number_of_crash_failed = 0
number_of_malicious_failed = 0
instance_list = []

if (number_of_random_failed+number_of_crash_failed+number_of_malicious_failed) > number_of_instances:
    raise ValueError("Number of unhealthy nodes cannnot sum to a number greater than the total number_of_instances")

index_list = list(range(number_of_instances))
random_failed_list = []
random_failed_list = []
crash_failed_list = []
malicious_failed = []
shuffle(index_list)

for i in range(number_of_random_failed):
    random_failed_list.append(index_list.pop(0))
    
for i in range(number_of_crash_failed):
    crash_failed_list.append(index_list.pop(0))
    
for i in range(number_of_malicious_failed):
    malicious_failed.append(index_list.pop(0))
    
#random_node
#crash_node    
#malicious_node    

print("Loading database...")
data_handler = DataProcessing(
    path = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv",
    total_lines = dataset_size,
    delimiter_char = ",",
    target_column = "Rating")
    
dataset_partition_size = int(numpy.floor(999/(turns*number_of_instances)))
print("dataset_partition_size =",dataset_partition_size)    
    
for i in range(number_of_instances):
    #Assumes that an index never gonna be in multiple lists at the same time
    random_node     = i in random_failed_list
    crash_node      = i in crash_failed_list
    malicious_node  = i in malicious_failed

    factorization_machine = FactorizationMachine(
        iterations                      = 20,
        learning_rate                   = 1/(100),
        latent_vectors                  = 4,
        regularization                  = 1/(1000),
        slice_size                      = 20,
        batch_size                      = 20,
        slice_patience                  = 5,
        iteration_patience              = 5,
        slice_patience_threshold        = 0.0000001,
        iteration_patience_threshold    = 0.0000001,
        name                            = str(i),
        random_failed                   = random_node,
        crash_failed                    = crash_node,
        malicious_failed                = malicious_node)
        
    instance_list.append(factorization_machine)
    
correction_offset = dataset_size - dataset_partition_size*turns*number_of_instances
sample_end = dataset_partition_size + correction_offset

print("About to start iterations through the dataset.")    
start = time.time()   
for i in range(turns):

    weight_matrices = []

    for j in range(number_of_instances):
        print("Turn: ",i,"Node: ",j)
        trainX,trainY,validationX,validationY,validation_indexes = data_handler.process_csv_data(lineStart = sample_start, lineEnd = sample_end)
    
        instance_list[j].learn(trainX,trainY)
        rmse = instance_list[j].predict(validationX,validationY)
        print('{"metric": "RMSE '+instance_list[j].name+'", "value": '+str(numpy.round(rmse,5))+'}')
        
        weight_matrices.append(instance_list[j].model)
    
        sample_start = sample_start + dataset_partition_size
        sample_end = sample_end + dataset_partition_size
        
    tardigrade_matrices = numpy.array(weight_matrices)    
    #print("tardigrade_matrices.shape",tardigrade_matrices.shape)
        
    for j in range(number_of_instances):
        #numpy.delete creates a new list without the instance_list[j] model
        instance_list[j].tardigrade(data_handler,numpy.delete(tardigrade_matrices,j,axis=0))

end = time.time()    
            

print(int((end - start)/60)," Minutes")
print((((end - start)/60)-int((end - start)/60))*60," Seconds")
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
