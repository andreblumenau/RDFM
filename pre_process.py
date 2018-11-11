import numpy
from numpy import genfromtxt
import random as random
from random import randint

def delete_column(array, *args):
    filtered_names = [x for x in array.dtype.names if x not in args]
    return array[filtered_names]

def process_csv_data(path, lineStart,lineEnd,delimiter_char,target_column):
#list(argh.dtype.names)
    read_names = open(path,'r')
    lines_for_names = read_names.readlines()[0:1]
    read_names.close()
    names_table = genfromtxt(lines_for_names, delimiter=delimiter_char, names=True)
    column_names = names_table.dtype.names

    big_file = open(path,'r')
    lines = big_file.readlines()[lineStart:lineEnd]
    big_file.close()
    seventyPercent = int((lineEnd-lineStart)*0.7)
    one_hundred_percent = lineEnd-lineStart
    
    training_lines = lines[0:seventyPercent]
    training   = genfromtxt(training_lines, delimiter=delimiter_char, names=True)
    training_target = training[target_column].copy()
    training_target = training_target.view(numpy.float64).reshape(training_target.size,1)  

  
    training_features =  [[float(y) for y in x] for x in training]
    training_features = numpy.delete(training_features,-1,axis=1)
    
    training_features = numpy.clip(training_features,0.0, 1.0)    
    
    validation_lines = lines[seventyPercent+1:one_hundred_percent]
    validation_lines = [lines[0]]+validation_lines
    validation = genfromtxt(validation_lines, delimiter=delimiter_char, names=True)
    validation_target = validation[target_column].copy()
    validation_target =  validation_target.view(numpy.float64).reshape(validation_target.size,1)
    
    validation_features =  [[float(y) for y in x] for x in validation]
    validation_features = numpy.delete(validation_features,-1,axis=1)
    validation_features = numpy.clip(validation_features,0.0, 1.0)

    training_bias_vector = numpy.tile(1,(training_features.shape[0],1))
    validation_bias_vector = numpy.tile(1,(validation_features.shape[0],1))
    
    training_features = numpy.hstack((training_bias_vector,training_features))
    validation_features = numpy.hstack((validation_bias_vector,validation_features))    
    
    return training_features,training_target,validation_features,validation_target