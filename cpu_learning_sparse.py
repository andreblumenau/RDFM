import numpy
import array
###########################
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, hstack,vstack

class CPULearning:
    def __init__(self,iterations, alpha, regularization,batch_size,iteration_patience,iteration_patience_threshold):
        self.iterations                   = iterations                   
        self.alpha                        = alpha                        
        self.regularization               = regularization               
        self.batch_size                   = batch_size                   
        self.iteration_patience           = iteration_patience           
        self.iteration_patience_threshold = iteration_patience_threshold

    def optimize(self,training_features, training_targets,weight_matrix):
        N = training_features.shape[0]
        M = weight_matrix.shape[1]
        
        tensor_of_x_features = []
        tensor_of_x_squared  = []
        tensor_of_x_features_squared=[]
    
        matrix_set_diag_to_zero = numpy.tile(1.0,(training_features.shape[1],training_features.shape[1]))
        numpy.fill_diagonal(matrix_set_diag_to_zero,0.0)
    
        historical_gradient = numpy.tile(0.0,(weight_matrix.shape))
    
        for i in range(N):
            tensor_of_x_features.append(training_features[i])
            tensor_of_x_squared.append(training_features[i].multiply(training_features[i]))#csr_matrix.transpose(training_features[i]).dot(training_features[i]))
        
        for i in range(N):        
            tensor_of_x_squared[i].setdiag(0)
            tensor_of_x_squared[i] = tensor_of_x_squared[i]
            matrix = tensor_of_x_features[i].copy()
            matrix.data **=2            
            tensor_of_x_features_squared.append(matrix)
        
        tensor_of_proto_vx     = []
        tensor_of_proto_square = []
        vrau                   = []
        b = []        
        
        for i in range(N):
            tensor_of_proto_vx.append(sparse.eye(2, dtype=numpy.float32)) #= array.array('i',(0,)*N)
            tensor_of_proto_square.append(sparse.eye(2, dtype=numpy.float32)) #= array.array('i',(0,)*N)
            vrau.append(sparse.eye(2, dtype=numpy.float32))
            b.append(0)
            
        vector_of_prediction = numpy.tile(0.0,N)
        vector_of_sum = numpy.tile(1.0,(M,1))
        vector_of_gradient = numpy.tile(0.0,N)
        
        weight_matrix_square = numpy.tile(0.0,(weight_matrix.shape))
        update_step = numpy.tile(0.0,(weight_matrix.shape))
    
        batch_count = numpy.floor(N/self.batch_size).astype(numpy.int32)
        seed = 0
        
        #idxs = numpy.linspace(0,self.batch_size,N,dtype=numpy.int32)  
    
        patience_counter = 0
        last_iteration_error = 0
    
        #error_iter_array = numpy.tile(1,(iterations,1))
        error_iter_array = numpy.empty(self.iterations, dtype=numpy.float32)
    
        for i in range(self.iterations):
            seed = seed + 1
            numpy.random.seed(seed)
            random_idx_list = numpy.random.permutation(N)
    
            idxs = 0
            init = 0
            ending = 0
            error_sum = 0        
            
            for j in range(batch_count):
                init = j*self.batch_size
                ending = min(len(random_idx_list),(j+1)*self.batch_size)
    
                idxs = random_idx_list[init:ending]            
                weight_matrix[numpy.abs(weight_matrix)<0.0000001]=0 
                weight_matrix_square = weight_matrix*weight_matrix
                
                for k in idxs:
                    tensor_of_proto_vx[k]=tensor_of_x_features[k].dot(weight_matrix)
                    #print("tensor_of_proto_vx[k].shape = ",tensor_of_proto_vx[k].shape)#(2,2)
                    tensor_of_proto_square[k]=(tensor_of_x_features_squared[k].dot(weight_matrix_square))
                    vector_of_prediction[k]=((tensor_of_proto_vx[k]*tensor_of_proto_vx[k])- tensor_of_proto_square[k]).dot(vector_of_sum).sum(axis=1)*0.5
                
                vector_of_prediction = vector_of_prediction.reshape(len(vector_of_prediction),1)
                
                for k in idxs:
                    b[k] =  numpy.asscalar(training_targets[k]-vector_of_prediction[k])
    
                b = numpy.array(b)
    
                error_sum = error_sum+b.mean()
                #print(b.mean())
                
                
                vector_of_gradient = -2*b
                for k in idxs:
                    vrau[k] = tensor_of_x_squared[k].dot(weight_matrix)
                
                updates = None
                
                for k in idxs:
                    if updates is not None:
                        updates = updates+(vector_of_gradient[k].T*vrau[k].T).T
                    else:
                        updates = (vector_of_gradient[k].T*vrau[k].T).T
                    
                update_step = updates+weight_matrix_square*self.regularization
        
                #ADAGRAD UPDATE
                historical_gradient += numpy.multiply(update_step,update_step)
                weight_matrix -= self.alpha/(numpy.sqrt(historical_gradient)) * update_step
                #weight_matrix -= self.alpha/(numpy.multiply((numpy.sqrt(historical_gradient)),update_step)+0.000001)           
    
            error_iter_array[i] = error_sum/batch_count
    
            if numpy.abs(numpy.abs(error_iter_array[i]) - last_iteration_error) < self.iteration_patience_threshold:
                patience_counter = patience_counter+1
            else:
                patience_counter = 0 #RESET
            
            if patience_counter == self.iteration_patience:
                break #
            
            last_iteration_error = numpy.abs(error_iter_array[i])
            
        return weight_matrix,error_iter_array.mean(),error_iter_array#return array with the most errors
