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
from cpu_learning import optimize
import csv
import gc

class FactorizationMachine:
    def get_random_weight_matrix(self,number_of_features,number_of_latent_vectors):
        model =  numpy.random.ranf((number_of_features, number_of_latent_vectors))
        model = model / numpy.sqrt((model*model).sum())
        return model
    
    def __init__(self,iterations,learning_rate,latent_vectors,regularization,slice_size,batch_size,
    slice_patience,iteration_patience,slice_patience_threshold,iteration_patience_threshold):    

        if slice_size < batch_size:
            raise ValueError('"slice_size" parameter cannot be smaller than "batch_size" parameter.')
            
        if iteration_patience >= iterations:
            raise ValueError('"iteration_patience" parameter cannot be smaller than "iterations" parameter.')                      

        #"Private" properties
        self.model = None
        
        #Parameterized properties
        self.iterations                     = iterations
        self.learning_rate                  = learning_rate               
        self.latent_vectors                 = latent_vectors              
        self.regularization                 = regularization              
        self.slice_size                     = slice_size                 
        self.batch_size                     = batch_size                  
        self.slice_patience                 = slice_patience             
        self.iteration_patience             = iteration_patience              
        self.slice_patience_threshold       = slice_patience_threshold   
        self.iteration_patience_threshold   = iteration_patience_threshold

    def learn(self,trainX,trainY):
    
        skip = 0
        end = 0   
        patience_counter = 0
        iteration_error = 0
        last_iteration_error = 0
    
        slice_count =  numpy.floor(trainX.shape[0]/self.slice_size).astype(numpy.int32)
        
        if self.slice_patience >= slice_count:
            raise ValueError('"slice_size" parameter cannot be smaller than "batch_size" parameter.')            
        
        if self.model is None:
            self.model = self.get_random_weight_matrix(trainX.shape[1],self.latent_vectors)
            
        for j in range(1):#(slice_count):        
            skip = j*self.slice_size    
            end = ((j+1)*self.slice_size)      
            self.model,iteration_error,error_iter_array = optimize( 
                trainX[skip:end], 
                trainY[skip:end], 
                iterations                   = self.iterations,
                alpha                        = self.learning_rate,
                regularization               = self.regularization,
                weight_matrix                = self.model,
                batch_size                   = self.batch_size,
                iteration_patience           = self.iteration_patience,            
                iteration_patience_threshold = self.iteration_patience_threshold)
        
            if numpy.abs(numpy.abs(iteration_error)-last_iteration_error) < self.slice_patience_threshold:
                patience_counter = patience_counter+1
            else:
                patience_counter = 0
                
            if patience_counter == self.slice_patience:
                break;
            
            last_iteration_error = numpy.abs(iteration_error)
            

            gc.collect()            
            
# f = open('errorPerIteration.csv','a',newline='')
# writer = csv.writer(f, delimiter=csv_delimiter)    

# for k in range(len(error_iter_array)):
    # writer.writerow([j,k,error_iter_array[k]])
# f.close()                
    
#end = time.time()

factorization_machine = FactorizationMachine(
    iterations                      = 10,
    learning_rate                   = 1/(100),
    latent_vectors                  = 4,
    regularization                  = 1/(1000),
    slice_size                      = 2,
    batch_size                      = 2,
    slice_patience                  = 5,
    iteration_patience              = 5,
    slice_patience_threshold        = 0.0000001,
    iteration_patience_threshold    = 0.0000001)
            
path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)            
factorization_machine.learn(trainX,trainY)
            
#print((end - start)," Seconds")
#print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,factorization_machine.model)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
