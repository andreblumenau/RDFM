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

class FactorizationMachine:

    self.model = None
    self.latent_vectors=0
    self.sample_size=0
    self.iterations=0
    self.regularization=0
    self.patience=0
    self.iteration_error_diff_lower_limit=0
    self.batch_size=0

    def get_random_model(number_of_features,number_of_latent_vectors):
        #modelo =  numpy.random.ranf((trainX.shape[1], number_of_latent_vectors))
        modelo =  numpy.random.ranf((number_of_features, number_of_latent_vectors))
        modelo = modelo / numpy.sqrt((modelo*modelo).sum())
        return modelo
    
    def function(self,model=None,features,latent_vectors,sample_size,iterations,regularization,
    patience,iteration_error_diff_lower_limit,batch_size):    

        if sample_size < batch_size:
            raise ValueError('sample_size parameter cannot be smaller than batch_size parameter.')
    
        if model is None:
            self.model = get_random_model(features,latent_vectors)
        else:
            self.model = model
            
        self.latent_vectors = latent_vectors
        self.sample_size = sample_size
        self.iterations = iterations
        self.regularization = regularization
        self.patience = patience
        self.iteration_error_diff_lower_limit = iteration_error_diff_lower_limit
        self.batch_size = batch_size
        

        
    
    #take - Defines the number of events that will be kept on memory simultaneously
    #memory_split - How many parts the will be, depends on "take"
    take = 2
    memory_split =  numpy.floor(trainX.shape[0]/take).astype(numpy.int32)
    
    # modelo =  numpy.random.ranf((trainX.shape[1], latent_vectors))
    # modelo = modelo / numpy.sqrt((modelo*modelo).sum())
    
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
        
    start = time.time()    
    #for j in range(memory_split):    
    for j in range(1):
        skip = j*take    
        end = ((j+1)*take)      
        modelo,iteration_error,error_iter_array = optimize( 
            trainX[skip:end], 
            trainY[skip:end], 
            iterations=5, alpha=1/(100),
            regularization=1/(1000),
            weight_matrix=modelo,
            patience=10,
            batch_size=2,
            iteration_error_diff_lower_limit=0.0000001)
    
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
            writer.writerow([j,k,error_iter_array[k]])
        f.close()    
        gc.collect()
    
end = time.time()

print((end - start)," Seconds")
print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
