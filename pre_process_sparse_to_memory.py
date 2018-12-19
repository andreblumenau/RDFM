import csv
import numpy
import time
from numpy import genfromtxt
from numpy import random
from random import shuffle
import gc
###########################
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, hstack,vstack
###########################
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)
###########################

class DataProcessing:
    def __init__(self,path,total_lines,delimiter_char,target_column,segment_size=100):
        self.path = path
        self.delimiter = delimiter_char

        #Better code can be written to read the first line(headers) of a csv file.
        csv_file = open(self.path,"r")
        reader = csv.reader(csv_file,delimiter=self.delimiter)

        header_line_index_list = [0]
        headers = self.read_my_lines(reader, header_line_index_list)
        header_line=[]
        for item in headers:
            #Skips first column
            header_line.append(item[1:])
        csv_file.close()
    
        #Higly inneficient but used only once.
        self.index_for_target_column = header_line[0].index(target_column)

        self.indexes = list(range(1,int(total_lines)))
        self.random_indexes = list(range(0,int(total_lines-1)))
        #numpy.random.shuffle(self.indexes)
        random.shuffle(self.random_indexes)
        
        self.file_to_memory(segment_size)

    def delete_column(self,array, *args):
        filtered_names = [x for x in array.dtype.names if x not in args]
        return array[filtered_names]

    def process_csv_data(self,lineStart,lineEnd):
        seventyPercent = int((lineEnd-lineStart)*0.7)
        
        training_indexes = self.random_indexes[lineStart:(lineStart+seventyPercent)]
        training_features,training_target = self.features_and_target_from_indexes(training_indexes)  
        
        validation_indexes = self.random_indexes[(lineStart+seventyPercent+1):lineEnd]
        validation_features, validation_target = self.features_and_target_from_indexes(validation_indexes)   
        
        return training_features,training_target,validation_features,validation_target,validation_indexes
    
    def features_and_target_from_indexes(self,indexes):
        #print("self.in_memory_dataset.shape = ",self.in_memory_dataset.shape)
        table_dense = self.in_memory_dataset[indexes].todense()
        #print("table_dense.shape = ",table_dense.shape)
        
        target = table_dense[:,self.index_for_target_column].copy()
        target = target.view(numpy.float64).reshape(target.size,1)  
    
        features = table_dense.copy()
        #[[print(y) for y in x] for x in table_dense[1:2,1:5]]
        #features =  [[float(y) for y in x] for x in table_dense]
        features = numpy.delete(features,-1,axis=1)
        
        features = numpy.clip(features,0.0, 1.0)    
        features_bias_vector = numpy.tile(1,(features.shape[0],1))
        features = numpy.hstack((features_bias_vector,features))

        target=sparse.csr_matrix(target)        
        features=sparse.csr_matrix(features)      

        return features,target
        
    def file_to_memory(self,segment_size):
        #TODO: CORRIGIR TAKE
        segment_count = int(numpy.ceil(len(self.indexes)/segment_size))
        self.in_memory_dataset = None     

        skip=0
        start_reading = time.time()
        #file = open(self.path,"rb")        
        #with open(self.path,"rb") as file: 
        for i in range(segment_count):
            file = open(self.path,"rb")
            #file = open(self.path,"rb")
            skip = i*segment_size
            #print("self.indexes[skip:(skip+segment_size)] = ",self.indexes[skip:(skip+segment_size)])
            file_generator = self.read_my_lines_sparse_numpy(file,self.indexes[skip:(skip+segment_size)])
            
            #print("skip = ",skip)
            #print("take = ",(skip+segment_size))
            if self.in_memory_dataset is None:        
                temp = numpy.loadtxt(file_generator,delimiter=",")
                self.in_memory_dataset = scipy.sparse.csr_matrix(temp)
            else:
                #print("i =",i)
                #print("self.in_memory_dataset.shape =",self.in_memory_dataset.shape)
                temp = numpy.loadtxt(file_generator,delimiter=",")
                temp = scipy.sparse.csr_matrix(temp)
                #print("temp.shape =",temp.shape)
                self.in_memory_dataset = scipy.sparse.vstack((self.in_memory_dataset,temp))
            file.close()
            gc.collect()
        
        file.close()
        gc.collect()

    def read_my_lines(self,csv_reader, lines_list):
        # make sure every line number shows up only once:
        lines_set = set(lines_list)
        for line_number, row in enumerate(csv_reader):
            if line_number in lines_set:
                yield row
                lines_set.remove(line_number)
                # Stop when the set is empty
                if not lines_set:
                    break        
        
    def read_my_lines_sparse_numpy(self,file, lines_list):
        list_of_floats = []
        
        if type(lines_list).__module__==numpy.__name__:
            lines_list = lines_list.tolist()
        
        if 0 in lines_list:
            index_of_zero = lines_list.index(0)
            lines_list.remove(0)
        
        for line_number, row in enumerate(file):
            if line_number == lines_list[0]:
                #print(line_number)            
                #yield [ float(i) for i in row ]
                #print(row)
                yield row#[ i for i in row ]
                lines_list.pop(0)
                # Stop when the set is empty
                if not lines_list:
                    break                                        