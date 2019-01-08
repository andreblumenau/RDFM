import numpy
import time
import pre_process_sparse_to_memory
import factorization_machine_sparse
from factorization_machine_sparse import FactorizationMachine
from pre_process_sparse_to_memory import DataProcessing
from metrics_sparse import evaluate
import csv
import random
from random import shuffle

dataset_size = 1000000    
sample_start = 0
sample_end = 0
turns = 100
number_of_instances = 6
number_of_random_failed = 0
number_of_crash_failed = 0
number_of_malicious_failed = 0
instance_list = []

if (number_of_random_failed+number_of_crash_failed+number_of_malicious_failed) > number_of_instances:
    raise ValueError("Number of unhealthy nodes cannnot sum to a number greater than the total number of instances")

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

print("Loading database...")
data_handler = DataProcessing(
    path = '/floyd/input/movielens/movielens.csv',
    total_lines = dataset_size,
    delimiter_char = ",",
    target_column = "Rating",
    segment_size=10000
)
    
dataset_partition_size = int(numpy.floor((1000000-1)/(turns*number_of_instances)))
print("dataset_partition_size =",dataset_partition_size)    
    
for i in range(number_of_instances):
    #Assumes that an index never gonna be in multiple lists at the same time
    random_node     = i in random_failed_list
    crash_node      = i in crash_failed_list
    malicious_node  = i in malicious_failed

    factorization_machine = FactorizationMachine(
        iterations                      = 5,#Per Batch
        learning_rate                   = 1/(1000),
        latent_vectors                  = 10,
        regularization                  = 1/(1000),
        slice_size                      = dataset_partition_size, #15000,#RAM BUFFER FOR MATRICES
        batch_size                      = 10,  #LEARNING MINI-BATCH-SIZE
        slice_patience                  = 1,
        iteration_patience              = 4,
        slice_patience_threshold        = 0.000001,
        iteration_patience_threshold    = 0.00001,
        #slice_patience_threshold        = 0.0000001,
        #iteration_patience_threshold    = 0.0000001,
        name                            = str(i),
        random_failed                   = random_node,
        crash_failed                    = crash_node,
        malicious_failed                = malicious_node)
        
    instance_list.append(factorization_machine)
    
correction_offset = dataset_size - dataset_partition_size*turns*number_of_instances
sample_end = dataset_partition_size + correction_offset #This correction tends to turn the first sample smaller than the dataset_partition_size

print("About to start iterations through the dataset.")    
start = time.time()   
rmse_str ="None"
for i in range(turns):

    weight_matrices = []

    for j in range(number_of_instances):
        rmse_str = "None"
        print("Turn: ",i,"Node: ",j)
        
        if not instance_list[j].crash_failed:
            trainX,trainY,validationX,validationY,validation_indexes = data_handler.process_csv_data(lineStart = sample_start, lineEnd = sample_end)
            print("Successful read from dataset.")
            instance_list[j].learn(trainX,trainY)
            rmse = instance_list[j].predict(validationX,validationY,error_buffer=25)
            rmse_str = str(numpy.round(rmse,5))
            
        print('{"metric": "RMSE '+instance_list[j].name+'", "value": '+rmse_str+'}')
        
        weight_matrices.append(instance_list[j].model)
        print("read ",sample_end," from ",dataset_size,"(",(sample_end/dataset_size)*100,"%)")
    
        sample_start = sample_start + dataset_partition_size
        sample_end = sample_end + dataset_partition_size
        
    tardigrade_matrices = numpy.array(weight_matrices)    
        
    for j in range(number_of_instances):
        #numpy.delete creates a new list without the instance_list[j] model (removes FM own model so it wont be considered 2 times)
        #instance_list[j].tardigrade(data_handler,numpy.delete(tardigrade_matrices,j,axis=0))
        instance_list[j].tardigrade(data_handler,tardigrade_matrices,top_n_models = 4)

end = time.time()    
            

print(int((end - start)/60)," Minutes")
print((((end - start)/60)-int((end - start)/60))*60," Seconds")