import numpy
import cpu_learning_sparse
from cpu_learning_sparse import CPULearning
import gc
from pre_process_sparse import DataProcessing #Talvez desnecessário
from metrics_sparse import evaluate
from metrics_sparse import evaluate_rmse

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
        self.optimization_routine = CPULearning(
            iterations                   = iterations,
            alpha                        = learning_rate,
            regularization               = regularization,
            batch_size                   = batch_size,
            iteration_patience           = iteration_patience,
            iteration_patience_threshold = iteration_patience_threshold
        )
        
        #Parameterized properties
        self.latent_vectors                 = latent_vectors              
        self.slice_size                     = slice_size                 
        self.slice_patience                 = slice_patience             
        self.slice_patience_threshold       = slice_patience_threshold   

    def learn(self,trainX,trainY):
    
        skip = 0
        end = 0   
        patience_counter = 0
        iteration_error = 0
        last_iteration_error = 0
    
        slice_count =  numpy.floor(trainX.shape[0]/self.slice_size).astype(numpy.int32)
        print("slice_count = ",slice_count)
        
        if self.slice_patience >= slice_count:
            raise ValueError('"slice_patience" parameter cannot be smaller than "slice_count" parameter.')            
        
        if self.model is None:
            self.model = self.get_random_weight_matrix(trainX.shape[1],self.latent_vectors)
            
        for j in range(slice_count):#(slice_count):        
            skip = j*self.slice_size    
            end = ((j+1)*self.slice_size)      
            self.model,iteration_error,error_iter_array = self.optimization_routine.optimize( 
                training_features            = trainX[skip:end], 
                training_targets             = trainY[skip:end], 
                weight_matrix                = self.model
            )
        
            if numpy.abs(numpy.abs(iteration_error)-last_iteration_error) < self.slice_patience_threshold:
                patience_counter = patience_counter+1
            else:
                patience_counter = 0
                
            if patience_counter == self.slice_patience:
                break;
            
            last_iteration_error = numpy.abs(iteration_error)

            gc.collect()

    def predict(self,validationX,validationY,error_buffer=5):
        rmse,error_by_index = evaluate(validationX,validationY,self.model)
        
        if error_buffer > error_by_index.shape[0]:
            error_buffer = error_by_index.shape[0]
        
        self.smallest_error = error_by_index[0:error_buffer,1].astype(numpy.int32)
        self.greatest_error = error_by_index[(len(error_by_index)-error_buffer):len(error_by_index),1].astype(numpy.int32)
        self.error_buffer = error_buffer
        #print("self.smallest_error",self.smallest_error)
        #print("self.smallest_error.shape",self.smallest_error.shape)
        
        return rmse
            
    def tardigrade(self,data_handler,neighbourhood_models):
        indexes = numpy.hstack((self.smallest_error,self.greatest_error))        
        features,target = data_handler.features_and_target_from_indexes(indexes)
        index_and_rmse = numpy.tile(1,(neighbourhood_models.shape[0],2))
        
        for i in range(neighbourhood_models.shape[0]):
            index_and_rmse[i][1] = i
            index_and_rmse[i][0] = evaluate_rmse(features,target,neighbourhood_models[i])
        
        index_and_rmse = index_and_rmse[index_and_rmse[:,0].argsort()]
        tensor = numpy.tile(0,(1,self.model.shape[0],self.model.shape[1]))
        tensor[0] = self.model
        neighbourhood_models = neighbourhood_models[index_and_rmse[0:max(index_and_rmse.shape[0],self.error_buffer),1]]
        
        self.model = numpy.vstack((neighbourhood_models,tensor)).mean(axis=0)
        return
        