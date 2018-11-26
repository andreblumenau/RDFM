import numpy
import os
from numpy import genfromtxt


index = 0
lineStart = 0
lineEnd = 0
listLines = []
noncarriage = 0
lastCharacterWasNewLine= False
with open("C:\Gondolin\Bohemian Rhapsody.txt",'r',) as f:
    while True:
        #print("ué")
        c = f.read(1)#.decode('UTF-8')
        #print(c)
        index = index+1
        if not c:
            print("NOT C")
            break
        
        if '\n' in c or '\r' in c:  
            print("NEWLINE OR CARRIAGE RETURN")
            lastCharacterWasNewLine= True
        else:            
            if lastCharacterWasNewLine:
                diff =index-lineStart
                tuple=(lineStart,diff+1)
                listLines.append(tuple)
                lineStart=index+1       
                lastCharacterWasNewLine=False
            #noncarriage = noncarriage +1
            
f.close()

        #lineStart = index

print("End of line indexing.")

with open("C:\Gondolin\Bohemian Rhapsody.txt",'r') as f:
#with open("C:\PosGrad\Movielens1M\data_processed_ 1 .csv",'rb') as f:
#C:\PosGrad\Movielens1M\data_processed_ 1 .csv
    for i in range(len(listLines)):
        print("iteration = ",i," ######################################################################")
        item = listLines[i]
        f.seek(item[0],0)
        line = str(f.read(item[1]))
        print(line)

