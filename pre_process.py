import numpy
from numpy import genfromtxt
import random as random
from random import randint
#import linecache

class DataProcessing:
     def __init__(path,delimiter_char,target_column)
        self.path = path
        self.delimiter = delimiter_char
        self.target_column = target_column

    def delete_column(self,array, *args):
        filtered_names = [x for x in array.dtype.names if x not in args]
        return array[filtered_names]
    
    def process_csv_data(self,lineStart,lineEnd):
        big_file = open(self.path,'r')
        header_lines = big_file.readlines()[0]
        lines = big_file.readlines()[lineStart:lineEnd]
        big_file.close()
        seventyPercent = int((lineEnd-lineStart)*0.7)
        one_hundred_percent = lineEnd-lineStart
        
        training_lines = lines[lineStart:(lineStart+seventyPercent)]
        training   = genfromtxt(training_lines, delimiter=self.delimiter_char, names=True)
        training_target = training[self.target_column].copy()
        training_target = training_target.view(numpy.float64).reshape(training_target.size,1)  
    
    
        training_features =  [[float(y) for y in x] for x in training]
        training_features = numpy.delete(training_features,-1,axis=1)
        
        training_features = numpy.clip(training_features,0.0, 1.0)    
        
        validation_lines = lines[seventyPercent+1:one_hundred_percent]
        validation_lines = [header_lines[0]]+validation_lines
        validation = genfromtxt(validation_lines, delimiter=self.delimiter_char, names=True)
        validation_target = validation[self.target_column].copy()
        validation_target =  validation_target.view(numpy.float64).reshape(validation_target.size,1)
        
        validation_features =  [[float(y) for y in x] for x in validation]
        validation_features = numpy.delete(validation_features,-1,axis=1)
        validation_features = numpy.clip(validation_features,0.0, 1.0)
    
        training_bias_vector = numpy.tile(1,(training_features.shape[0],1))
        validation_bias_vector = numpy.tile(1,(validation_features.shape[0],1))
        
        training_features = numpy.hstack((training_bias_vector,training_features))
        validation_features = numpy.hstack((validation_bias_vector,validation_features))    
        
        return training_features,training_target,validation_features,validation_target