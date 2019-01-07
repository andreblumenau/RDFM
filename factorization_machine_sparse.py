import numpy
import cpu_learning_sparse
from cpu_learning_sparse import CPULearning
import gc
from pre_process_sparse import DataProcessing #Talvez desnecessário
from metrics_sparse import evaluate
from metrics_sparse import evaluate_rmse
import scipy
from scipy import sparse

class FactorizationMachine:
    def get_random_weight_matrix(self,number_of_features,number_of_latent_vectors):
        model =  numpy.random.ranf((number_of_features, number_of_latent_vectors))
        model = model / numpy.sqrt((model*model).sum())
        return model
    
    def __init__(self,iterations,learning_rate,latent_vectors,regularization,slice_size,batch_size,
    slice_patience,iteration_patience,slice_patience_threshold,iteration_patience_threshold,name="",
    random_failed=False,crash_failed=False,malicious_failed=False):    
        self.name=name
        self.random_failed    = random_failed
        self.crash_failed     = crash_failed
        self.malicious_failed = malicious_failed
        
        if self.random_failed    : self.name = self.name + " (RANDOM OUTPUT)"
        if self.crash_failed     : self.name = self.name + " (CRASH)"
        if self.malicious_failed : self.name = self.name + " (MALICIOUS)"
        
        if slice_size < batch_size:
            raise ValueError('"slice_size" parameter cannot be smaller than "batch_size" parameter.')
            
        if iteration_patience >= iterations:
            raise ValueError('"iteration_patience" parameter cannot be greater or equal to "iterations" parameter.')                      

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
        if self.random_failed:
            self.model = self.get_random_weight_matrix(trainX.shape[1],self.latent_vectors)
            return

        if self.crash_failed:
            self.model = None
            return        
    
        if self.malicious_failed:
            #Inverts target matrix to optmize for error
            trainX = scipy.sparse.csr_matrix(1-trainX.todense())
        
        skip = 0
        end = 0   
        patience_counter = 0
        iteration_error = 0
        last_iteration_error = 0
            
        print("trainX.shape[0] = ",trainX.shape[0])
        slice_count =  max(numpy.floor(trainX.shape[0]/self.slice_size).astype(numpy.int32),1)
        
        
        if self.slice_patience > slice_count:
            print("slice_count = ",slice_count)
            print("slice_patience = ",self.slice_patience)
            raise ValueError('"slice_count" parameter cannot be smaller than "slice_patience" parameter.')            
        
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

    def predict(self,validationX,validationY,error_buffer=50):
        if self.crash_failed: return None
        
        rmse,error_by_index = evaluate(validationX,validationY,self.model,name=self.name)
        
        if error_buffer > error_by_index.shape[0]:
            print("error_buffer > total errors, assumes error_buffer = ",error_by_index.shape[0])
            error_buffer = error_by_index.shape[0]
        
        self.smallest_error = error_by_index[0:error_buffer,1].astype(numpy.int32)
        self.greatest_error = error_by_index[(len(error_by_index)-error_buffer):len(error_by_index),1].astype(numpy.int32)
        self.error_buffer = error_buffer
        
        return rmse
            
    def tardigrade(self,data_handler,neighbourhood_models,top_n_models = 5):
        if self.crash_failed: return
        if self.malicious_failed: return
        if self.random_failed: return
        #TODO: CORRIGIR ISSO
        if len(neighbourhood_models) <= 1: return        
        
        boolean_array_of_is_none = [i is None for i in neighbourhood_models]
        
        while True in boolean_array_of_is_none:
            index_of_none = boolean_array_of_is_none.index(True)
            neighbourhood_models = numpy.delete(neighbourhood_models,index_of_none)
            boolean_array_of_is_none = [i is None for i in neighbourhood_models]
    
        indexes = numpy.hstack((self.smallest_error,self.greatest_error))        
        features,target = data_handler.features_and_target_from_indexes(indexes)
        index_and_rmse = numpy.tile(1,(neighbourhood_models.shape[0],2))
        
        for i in range(neighbourhood_models.shape[0]):
            index_and_rmse[i][1] = i
            index_and_rmse[i][0] = evaluate_rmse(features,target,neighbourhood_models[i])
        
        index_and_rmse = index_and_rmse[index_and_rmse[:,0].argsort()]
        tensor = numpy.tile(0,(1,self.model.shape[0],self.model.shape[1]))
        tensor[0] = self.model
        neighbourhood_models = neighbourhood_models[index_and_rmse[0:max(index_and_rmse.shape[0],top_n_models),1]]
        #TARDIGRADE OFF
        #neighbourhood_models = neighbourhood_models[index_and_rmse[0:index_and_rmse.shape[0],1]]
        
        self.model = (neighbourhood_models).sum(0)/len(neighbourhood_models)
        return
        