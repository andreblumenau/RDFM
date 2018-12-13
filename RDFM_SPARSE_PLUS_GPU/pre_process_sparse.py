import csv
import numpy
import time
from numpy import genfromtxt
from numpy import random
from random import shuffle
###########################
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, hstack,vstack

class DataProcessing:
    def __init__(self,path,total_lines,delimiter_char,target_column):
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

        self.random_sorted_indexes = list(range(1,int(total_lines)))
        #numpy.random.shuffle(self.random_sorted_indexes)
        random.shuffle(self.random_sorted_indexes)

    def delete_column(self,array, *args):
        filtered_names = [x for x in array.dtype.names if x not in args]
        return array[filtered_names]
    
    def features_and_target_from_indexes(self,indexes,buffer_size=5):
        start_reading = time.time()
        indexes.sort()
        #print(indexes)
        csv_file = open(self.path,"r")
        reader = csv.reader(csv_file,delimiter=self.delimiter)        
        
        file_generator = self.read_my_lines_sparse(reader,indexes)
        lines = []
        for item in file_generator:
            #Skips first column
            lines.append(item[1:])
        csv_file.close()
        
        shuffle(lines)
        table_dense = numpy.stack(lines)
        
        #table = table.tocsr()
                
        #print(table.shape)
        #print(table[1,:])
        
        target = table_dense[:,self.index_for_target_column].copy()
        target = target.view(numpy.float64).reshape(target.size,1)  
    
        features =  [[float(y) for y in x] for x in table_dense]
        features = numpy.delete(features,-1,axis=1)
        
        features = numpy.clip(features,0.0, 1.0)    
        features_bias_vector = numpy.tile(1,(features.shape[0],1))
        features = numpy.hstack((features_bias_vector,features))

        target=sparse.csr_matrix(target)        
        features=sparse.csr_matrix(features)      

        #print(features.shape)
        #print(target.shape)
        #print(features[1,:])        
        #print(features[1])
        
        end_reading = time.time()
        print(round((end_reading-start_reading)/60,2)," minutos.")
        #print(lines[0])
        return features,target
    
    def process_csv_data(self,lineStart,lineEnd):
        seventyPercent = int((lineEnd-lineStart)*0.7)
        
        training_indexes = self.random_sorted_indexes[lineStart:(lineStart+seventyPercent)]
        #print("training_indexes = ",training_indexes.shape)
        training_features,training_target = self.features_and_target_from_indexes(training_indexes)  
        
        validation_indexes = self.random_sorted_indexes[(lineStart+seventyPercent+1):lineEnd]
        validation_features, validation_target = self.features_and_target_from_indexes(validation_indexes)   
        
        return training_features,training_target,validation_features,validation_target,validation_indexes
        
    def read_my_lines(self,csv_reader, lines_list):
        # make sure every line number shows up only once:
        lines_set = set(lines_list)
        for line_number, row in enumerate(csv_reader):
            if line_number in lines_set:
                #yield row#line_number, row
                yield row
                lines_set.remove(line_number)
                # Stop when the set is empty
                if not lines_set:
                    break  
    
    # scipy.sparse.csr_matrix
    def read_my_lines_float(self,csv_reader, lines_list):
        # make sure every line number shows up only once:
        #lines_set = set(lines_list)
        for line_number, row in enumerate(csv_reader):
            #if line_number in lines_set:
            if line_number == lines_list[0]:
                #yield row#line_number, row
                yield [ float(i) for i in row ]
                #lines_set.remove(line_number)
                #lines_list.remove()
                lines_list.pop(0)
                # Stop when the set is empty
                if not lines_list:
                    break
                    
    def read_my_lines_sparse(self,csv_reader, lines_list):
        # make sure every line number shows up only once:
        #lines_set = set(lines_list)
        list_of_floats = []
        
        if type(lines_list).__module__==numpy.__name__:
            lines_list = lines_list.tolist()
        
        if 0 in lines_list:
            index_of_zero = lines_list.index(0)
            lines_list.remove(0)
        
        for line_number, row in enumerate(csv_reader):
            #if line_number in lines_set:
            if line_number == lines_list[0]:
                #yield row#line_number, row
                #list_of_floats = [ float(i) for i in row ]
                yield [ float(i) for i in row ]
                #yield scipy.sparse.csr_matrix(list_of_floats)
                #lines_set.remove(line_number)
                #lines_list.remove()
                lines_list.pop(0)
                # Stop when the set is empty
                if not lines_list:
                    break                    