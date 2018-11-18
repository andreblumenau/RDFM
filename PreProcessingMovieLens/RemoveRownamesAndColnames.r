library(sp)
library(beepr)
library(data.table)

filename = ""
b = fread("data_processed_ 1 .csv",sep = ",")
fwrite(b,file="data_processed_ 1 .csv",row.names=FALSE)

start_time <- Sys.time()
 for(i in 2:1000){
	filename = paste("data_processed_",i,".csv")
	b = fread(filename,sep = ",")
	fwrite(b,file=filename,row.names=FALSE,col.names=FALSE)
}
end_time <- Sys.time()
print("Elapsed time:")
print(end_time - start_time)

beep()
#
#