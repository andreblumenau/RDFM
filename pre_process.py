import csv
import numpy
from numpy import genfromtxt
from numpy import random

class DataProcessing:
    def __init__(self,path,delimiter_char,target_column):
        self.path = path
        self.delimiter = delimiter_char

        #Better code can be written to read the first line(headers) of a csv file.
        csv_file = open(self.path,"r")
        reader = csv.reader(csv_file,delimiter=delimiter_char)

        header_line_index_list = [0]
        headers = read_my_lines(reader, header_line_index_list)
        header_line=[]
        for item in headers:
            #Skips first column
            header_line.append(item[1:])
        csv_file.close()
    
        #Higly inneficient but used only once.
        self.index_for_target_column = header_line[0].index(target_column)

        self.random_sorted_indexes = list(range(1,len(self.lines)))
        numpy.random.shuffle(self.random_sorted_indexes)

    def delete_column(self,array, *args):
        filtered_names = [x for x in array.dtype.names if x not in args]
        return array[filtered_names]
    
    def features_and_target_from_indexes(self,indexes):
        csv_file = open(self.path,"r")
        reader = csv.reader(csv_file,delimiter=delimiter_char)        
        
        file_generator = read_my_lines_float(reader,lines_list)
        lines = []
        for item in file_generator:
            #Skips first column
            lines.append(item[1:])
        csv_file.close()
        
        table = numpy.stack(lines)
        
        
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
        
    def read_my_lines(csv_reader, lines_list):
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
    
    def read_my_lines_float(csv_reader, lines_list):
        # make sure every line number shows up only once:
        lines_set = set(lines_list)
        for line_number, row in enumerate(csv_reader):
            if line_number in lines_set:
                #yield row#line_number, row
                yield [ float(i) for i in row ]
                lines_set.remove(line_number)
                # Stop when the set is empty
                if not lines_set:
                    break