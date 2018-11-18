library(sp)
library(beepr)
library(data.table)

filename = ""
b = fread("data_processed_ 1 .csv",sep = ",", select = c("Timestamp") )
b_range = range(b)
min_timestamp = 956703954
max_timestamp = 1046454590

max_timestamp = max_timestamp - min_timestamp

start_time <- Sys.time()
 for(i in 1:1000){
	filename = paste("data_processed_",i,".csv")
	b = fread(filename,sep = ",")
	b[,"Timestamp"] = b[,"Timestamp"]-min_timestamp
	b[,"Timestamp"] = b[,"Timestamp"]/max_timestamp
	fwrite(b,file=filename)
}
end_time <- Sys.time()
print("Tempo de Execução:")
print(end_time - start_time)

print(paste0("TIMESTAMP (MIN) = ",min_timestamp))
print(paste0("TIMESTAMP (MAX) = ",max_timestamp))

beep()
#
#