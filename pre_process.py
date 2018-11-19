import numpy
from numpy import genfromtxt
from numpy import random
#import random as random
#from random import randint
#import linecache

class DataProcessing:
    def __init__(self,path,delimiter_char,target_column):
        self.path = path
        self.delimiter = delimiter_char
        self.target_column = target_column

        big_file = open(self.path,'r')
        self.lines = big_file.readlines()
        big_file.close()
        self.lines = numpy.asarray(self.lines)

        self.random_sorted_indexes = list(range(1,len(self.lines)))
        numpy.random.shuffle(self.random_sorted_indexes)

    def delete_column(self,array, *args):
        filtered_names = [x for x in array.dtype.names if x not in args]
        return array[filtered_names]
    
    def table_from_indexes(self,indexes):
        indexes = numpy.insert(indexes,0,int(0)).tolist()     
        table_lines = self.lines[indexes]
        table = genfromtxt(table_lines, delimiter=self.delimiter, names=True)
        return table
    
    def process_csv_data(self,lineStart,lineEnd):
        seventyPercent = int((lineEnd-lineStart)*0.7)
        
        training_indexes = self.random_sorted_indexes[lineStart:(lineStart+seventyPercent)]
        training = self.table_from_indexes(training_indexes)
        # training_indexes = numpy.insert(training_indexes,0,int(0)).tolist()
        # training_lines = self.lines[training_indexes]
        # training   = genfromtxt(training_lines, delimiter=self.delimiter, names=True)
        training_target = training[self.target_column].copy()
        training_target = training_target.view(numpy.float64).reshape(training_target.size,1)  
    
        training_features =  [[float(y) for y in x] for x in training]
        training_features = numpy.delete(training_features,-1,axis=1)
        
        training_features = numpy.clip(training_features,0.0, 1.0)    
        
        validation_indexes = self.random_sorted_indexes[(lineStart+seventyPercent+1):lineEnd]
        validation = self.table_from_indexes(validation_indexes)
        # validation_indexes = numpy.insert(validation_indexes,0,int(0)).tolist()     
        # validation_lines = self.lines[validation_indexes]
        # validation = genfromtxt(validation_lines, delimiter=self.delimiter, names=True)
        validation_target = validation[self.target_column].copy()
        validation_target =  validation_target.view(numpy.float64).reshape(validation_target.size,1)
        
        validation_features =  [[float(y) for y in x] for x in validation]
        validation_features = numpy.delete(validation_features,-1,axis=1)
        validation_features = numpy.clip(validation_features,0.0, 1.0)
    
        training_bias_vector = numpy.tile(1,(training_features.shape[0],1))
        validation_bias_vector = numpy.tile(1,(validation_features.shape[0],1))
        
        training_features = numpy.hstack((training_bias_vector,training_features))
        validation_features = numpy.hstack((validation_bias_vector,validation_features))    
        
        return training_features,training_target,validation_features,validation_target,validation_indexes