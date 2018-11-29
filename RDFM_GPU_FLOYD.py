import cupy
import time
import pre_process_gpu
import factorization_machine_gpu
from factorization_machine_gpu import FactorizationMachineGPU
from pre_process_gpu import DataProcessing
from metrics_gpu import evaluate
import csv
import cupy

dataset_size = 1000000/100 -1#1000000-1    
sample_start = 0
sample_end = 0
turns = 150
number_of_instances = 5
instance_list = []


data_handler = DataProcessing(
    path = '/floyd/input/movielens/movielens.csv',
    total_lines = dataset_size,    
    delimiter_char = ",",
    target_column = "Rating")       
print("Database loaded.")

dataset_partition_size = int(cupy.floor(dataset_size/(turns*number_of_instances)))
print("dataset_partition_size =",dataset_partition_size)

for i in range(number_of_instances):
    factorization_machine = FactorizationMachineGPU(
        iterations                      = 10,#Per Batch
        learning_rate                   = 1/(100),
        latent_vectors                  = 4,
        regularization                  = 1/(1000),
        slice_size                      = 20*(3),
        batch_size                      = 20*(3),
        slice_patience                  = 5,
        iteration_patience              = 3,
        slice_patience_threshold        = 0.0001,
        iteration_patience_threshold    = 0.001)
        
    instance_list.append(factorization_machine)

print("Factorization Machines created.")

correction_offset = dataset_size - dataset_partition_size*turns*number_of_instances
sample_end = dataset_partition_size + correction_offset
sample_end = int(sample_end)

print("About to start iterations through the dataset.")    
start = time.time()    
for i in range(turns):    
    weight_matrices = []

    for j in range(number_of_instances):
        print("Turn: ",i,"Node: ",j)
        trainX,trainY,validationX,validationY,validation_indexes = data_handler.process_csv_data(lineStart = sample_start, lineEnd = sample_end)
    
        instance_list[j].learn(trainX,trainY)
        rmse = instance_list[j].predict(validationX,validationY)
        
        weight_matrices.append(instance_list[j].model)
    
        sample_start = sample_start + dataset_partition_size
        sample_end = sample_end + dataset_partition_size
        
    tardigrade_matrices = cupy.array(weight_matrices)    
    print("tardigrade_matrices.shape",tardigrade_matrices.shape)
        
    for j in range(number_of_instances):
        #cupy.delete creates a new list without the instance_list[j] model
        instance_list[j].tardigrade(data_handler,cupy.delete(tardigrade_matrices,j,axis=0))

end = time.time()    
            
print((end - start)," Seconds")
print(((end - start)/60)," Minutes")