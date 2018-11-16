import numpy

def learning(training_features, training_targets, iterations, alpha, regularization,weight_matrix,patience_limit,error_diff_limit = 0.0000001,performance_splits=1):
    N = training_features.shape[0]
    M = weight_matrix.shape[1]
    
    tensor_of_x_features = numpy.tile(0.0,(N,1,training_features.shape[1]))
    tensor_of_x_squared = numpy.tile(0.0,(N,training_features.shape[1],training_features.shape[1]))

    matrix_set_diag_to_zero = numpy.tile(1.0,(training_features.shape[1],training_features.shape[1]))
    numpy.fill_diagonal(matrix_set_diag_to_zero,0.0)

    for i in range(N):
        tensor_of_x_features[i]=training_features[i]
        tensor_of_x_squared[i]=training_features[i].dot(training_features[i])

    historical_gradient=numpy.tile(0.0,(weight_matrix.shape))
    tensor_of_x_squared = tensor_of_x_squared*matrix_set_diag_to_zero
    tensor_of_x_features_squared = tensor_of_x_features*tensor_of_x_features
    
    tensor_of_proto_vx = numpy.tile(0.0,(N,1,M))
    tensor_of_proto_square = numpy.tile(0.0,(N,1,M))
    vector_of_prediction = numpy.tile(0.0,N)
    vector_of_sum = numpy.tile(1.0,(M,1))
    vector_of_gradient = numpy.tile(0.0,N)
    
    weight_matrix_square = numpy.tile(0.0,(weight_matrix.shape))
    update_step = numpy.tile(0.0,(weight_matrix.shape))

    taker = numpy.floor(N/performance_splits).astype(numpy.int32)
    seed = 0
    
    idxs = numpy.linspace(0,taker,taker,dtype=numpy.int32)  

    patience = 0
    last_iteration_error = 0

    error_iter_array = numpy.tile(1,(iterations,1))     

    for i in range(iterations):
        seed = seed + 1
        numpy.random.seed(seed)
        random_idx_list = numpy.random.permutation(N)

        idxs = 0
        init = 0
        ending = 0
        error_sum = 0
        
        for j in range(performance_splits):
            init = j*taker
            ending = (j+1)*taker

            idxs = random_idx_list[init:ending]
        
            weight_matrix[numpy.abs(weight_matrix)<0.0000001]=0 
            weight_matrix_square = weight_matrix*weight_matrix
            tensor_of_proto_vx = numpy.tensordot(tensor_of_x_features[idxs],weight_matrix,axes=1)
            tensor_of_proto_square = numpy.tensordot(tensor_of_x_features_squared[idxs],weight_matrix_square,axes=1)
            vector_of_prediction = numpy.tensordot(((tensor_of_proto_vx*tensor_of_proto_vx) - tensor_of_proto_square),vector_of_sum,axes=1).sum(axis=1)*0.5
            b = training_targets[idxs]-vector_of_prediction           
            
            print(b.mean())
   
            error_sum = error_sum+b.mean()
            
            vector_of_gradient = -2*b
            vrau = numpy.tensordot(tensor_of_x_squared[idxs],weight_matrix,axes=1)
            update_step = ((vector_of_gradient.T*vrau.T).T).sum(axis=0)+weight_matrix_square*regularization
    
            #ADAGRAD UPDATE
            historical_gradient += update_step * update_step
            weight_matrix -= alpha/(numpy.sqrt(historical_gradient)) * update_step#+0.000001            

        error_iter_array[i] = error_sum/performance_splits

        if numpy.abs(numpy.abs(error_iter_array[i]) - last_iteration_error) < error_diff_limit:
          patience = patience+1
        else:
          patience = 0        
          
        if patience == patience_limit:
          break
        
        last_iteration_error = numpy.abs(error_iter_array[i])
        
    return weight_matrix,error_iter_array.mean(),error_iter_array#return array with the most errors