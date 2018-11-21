import numpy
from numpy import genfromtxt
from numpy import random

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
    
    def features_and_target_from_indexes(self,indexes):
        indexes = numpy.insert(indexes,0,int(0)).tolist()     
        table_lines = self.lines[indexes]
        table = genfromtxt(table_lines, delimiter=self.delimiter, names=True)
        
        target = table[self.target_column].copy()
        target = target.view(numpy.float64).reshape(target.size,1)  
    
        features =  [[float(y) for y in x] for x in table]
        features = numpy.delete(features,-1,axis=1)
        
        features = numpy.clip(features,0.0, 1.0)    
        features_bias_vector = numpy.tile(1,(features.shape[0],1))
        features = numpy.hstack((features_bias_vector,features))  
        
        return features,target
    
    def process_csv_data(self,lineStart,lineEnd):
        seventyPercent = int((lineEnd-lineStart)*0.7)
        
        training_indexes = self.random_sorted_indexes[lineStart:(lineStart+seventyPercent)]
        training_features,training_target = self.features_and_target_from_indexes(training_indexes)  
        
        validation_indexes = self.random_sorted_indexes[(lineStart+seventyPercent+1):lineEnd]
        validation_features, validation_target = self.features_and_target_from_indexes(validation_indexes)   
        
        return training_features,training_target,validation_features,validation_target,validation_indexes