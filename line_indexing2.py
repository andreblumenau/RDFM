import csv
import numpy
from numpy import genfromtxt

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
                #raise StopIteration

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
                #raise StopIteration


lines_list = [500,2,501,50,300]


#fl = next(open("C:\PosGrad\Movielens1M\data_processed_ 1 .csv","r"))
f = open("C:\PosGrad\Movielens1M\data_processed_ 1 .csv","r")
reader = csv.reader(f)

headers = read_my_lines(reader,[0])
header_line=[]
for item in headers:
    header_line.append(item[1:])
    
index_for_target_column = header_line[0].index('Rating')

ololo = read_my_lines_float(reader,lines_list)
lines = []
#lines.append(fl[0])
#print(dir(ololo))
for item in ololo:
    lines.append(item[1:])
f.close()
#for row in reader:

#lines = [list(item) for item in lines]
#array = numpy.array(lines[1:], dtype='f')
#table = genfromtxt(lines, delimiter=",", names=False)
hell = numpy.stack(lines)
hell.shape