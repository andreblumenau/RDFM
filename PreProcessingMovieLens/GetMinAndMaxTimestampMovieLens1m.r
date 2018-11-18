library(sp)
library(beepr)
library(data.table)

filename = ""
b = fread("data_processed_ 1 .csv",sep = ",", select = c("Timestamp") )
b_range = range(b)
min_timestamp = b_range[1]
max_timestamp = b_range[2]

start_time <- Sys.time()
 for(i in 2:1000){
	filename = paste("data_processed_",i,".csv")
	b = fread(filename,sep = ",", select = c("Timestamp") )
	b_range = range(b)
	if(b_range[1]< min_timestamp){
		min_timestamp = b_range[1]
	}
	
	if(b_range[2]> max_timestamp){
		max_timestamp = b_range[2]
	}	
 }
 
end_time <- Sys.time()
print("Elapsed time:")
print(end_time - start_time)

print(paste0("TIMESTAMP (MIN) = ",min_timestamp))
print(paste0("TIMESTAMP (MAX) = ",max_timestamp))

beep()
#
#