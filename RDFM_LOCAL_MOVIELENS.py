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
   
path_csv = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv"
delimiter  = ","
target_column = "Rating"

trainX,trainY,validationX,validationY = process_csv_data(path_csv, 0,1000,delimiter,target_column)

class FactorizationMachine:
    self.model = None
    
    self.iterations=0
    self.learning_rate=0
    self.latent_vectors=0
    self.regularization=0
    self.slice_size=0
    self.batch_size=0
    self.slice_patience=0
    self.iteration_patience=0    
    self.iteration_patience_threshold=0
    self.slice_patience_threshold=0
    
    self.memory_split=0

    def get_random_weight_matrix(number_of_features,number_of_latent_vectors):
        #modelo =  numpy.random.ranf((trainX.shape[1], number_of_latent_vectors))
        modelo =  numpy.random.ranf((number_of_features, number_of_latent_vectors))
        modelo = modelo / numpy.sqrt((modelo*modelo).sum())
        return modelo
    
    def function(self,iterations,learning_rate,latent_vectors,regularization,slice_size,batch_size,
    slice_patience,iteration_patience,iteration_patience_threshold,slice_patience_threshold):    

        if slice_size < batch_size:
            raise ValueError('slice_size parameter cannot be smaller than batch_size parameter.')

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
        self.iteration_patience_threshold   = iteration_patience_threshold
        self.slice_patience_threshold       = slice_patience_threshold   
        
        #Computed properties
        
        ###TODO
        #How many parts the will be, depends on "slice_size"
        #self.memory_split =  numpy.floor(trainX.shape[0]/slice_size).astype(numpy.int32)

        
    # iteration_error = 0 

    # sample_error_limit = 0.0000001
    # patience_counter = 0
    # slice_patience = 80
    
    # csv_delimiter = '	'
    # my_file = Path("/path/to/file")
    # f = open('errorPerIteration.csv','w',newline='')
    # writer = csv.writer(f, delimiter=csv_delimiter)
    # writer.writerow(["SAMPLE","ITERATION","RMSE"])
    # f.close()

    def learn(trainX,trainY):
    
        skip = 0
        end = 0   
        patience_counter = 0
        iteration_error = 0
        last_iteration_error = 0
    
        slice_count =  numpy.floor(trainX.shape[0]/slice_size).astype(numpy.int32)
        
        #start = time.time()

        
            
        if model is None:
            self.model = get_random_weight_matrix(trainX.shape[1],self.latent_vectors)
            
        if model.slice_size  >

        #for j in range(slice_count):        
        for j in range(1):
            skip = j*self.slice_size    
            end = ((j+1)*self.slice_size)      
            modelo,iteration_error,error_iter_array = optimize( 
                trainX[skip:end], 
                trainY[skip:end], 
                iterations                   = self.iterations,
                alpha                        = self.learning_rate,
                regularization               = self.regularization,
                weight_matrix                = self.modelo,
                batch_size                   = self.batch_size,
                iteration_patience           = self.iteration_patience,            
                iteration_patience_threshold = self.iteration_patience_threshold)
        
            if numpy.abs(numpy.abs(iteration_error)-last_iteration_error) < sample_error_limit:
                patience_counter = patience_counter+1
            else:
                patience_counter = 0
                
            if patience_counter == slice_patience:
                break;
            
            last_iteration_error = numpy.abs(iteration_error)
            
            # f = open('errorPerIteration.csv','a',newline='')
            # writer = csv.writer(f, delimiter=csv_delimiter)    
            
            # for k in range(len(error_iter_array)):
                # writer.writerow([j,k,error_iter_array[k]])
            # f.close()    
            gc.collect()
    
#end = time.time()

        # modelo,iteration_error,error_iter_array = optimize( 
            # trainX[skip:end], 
            # trainY[skip:end], 
            # iterations=5, alpha=1/(100),
            # regularization=1/(1000),
            # weight_matrix=modelo,
            # iteration_patience=10,
            # batch_size=2,
            # iteration_patience_threshold=0.0000001)


#print((end - start)," Seconds")
#print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,modelo)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
