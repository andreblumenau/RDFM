import numpy
import time
import winsound
import pre_process
import factorization_machine
from factorization_machine import FactorizationMachine
from pre_process import process_csv_data
from metrics import evaluate
import csv

factorization_machine = FactorizationMachine(
    iterations                      = 10,
    learning_rate                   = 1/(100),
    latent_vectors                  = 4,
    regularization                  = 1/(1000),
    slice_size                      = 2,
    batch_size                      = 2,
    slice_patience                  = 5,
    iteration_patience              = 5,
    slice_patience_threshold        = 0.0000001,
    iteration_patience_threshold    = 0.0000001)
            
trainX,trainY,validationX,validationY = process_csv_data(
    path = "C:\PosGrad\Movielens1M\data_processed_ 1 .csv",
    lineStart  = 0,
    lineEnd    = 1000,
    delimiter= ",",
    target_column="Rating") 
    
factorization_machine.learn(trainX,trainY)
            
#print((end - start)," Seconds")
#print(((end - start)/60)," Minutes")
evaluate(validationX,validationY,factorization_machine.model)
winsound.PlaySound("C:\\Users\\AndréRodrigo\\Downloads\\LTTP\\LTTP_Get_HeartPiece_StereoR.wav", winsound.SND_FILENAME)
