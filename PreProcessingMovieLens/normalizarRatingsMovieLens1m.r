library(sp)
library(beepr)
#filename = ""
start_time <- Sys.time()
 for(i in 1:1000){
	filename = paste("data_processed_",i,".csv")
	a = read.csv(filename,sep=",")
	a[,"Rating"] = a[,"Rating"]/5
	write.csv(a,file=filename)
 }
 
end_time <- Sys.time()
print("Elapsed Time:")
print(end_time - start_time)
beep()
#
#