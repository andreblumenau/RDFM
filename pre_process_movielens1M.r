library(dplyr)
library(slam)
library(Matrix)
library(beepr)

movies  <- "movies.dat" #Path to MOVIELENS 1M movies file
ratings <- "ratings.dat" #Path to MOVIELENS 1M ratings file
users_load   <- "users.dat" #Path to MOVIELENS 1M users file

#Ratings Table Structure:UserID::MovieID::Rating::Timestamp
ratings_table <-read.table(ratings,sep='=',fill=TRUE, quote="", encoding="UTF-8")
#Movies Table Structure: MovieID::Title::Genres
movies_table <-read.table(movies,sep='=', fill=TRUE, quote="", encoding="UTF-8")
#Users Table Structure: UserID::Gender::Age::Occupation::Zip-code
users_table <-read.table(users_load,sep='=',fill=TRUE, quote="", encoding="UTF-8")

head(movies_table,n=1)
head(ratings_table,n=1)
head(users_table,n=1)

#Movies
movies_id <- as.matrix(movies_table %>% distinct(V1))
movies <- movies_table %>% distinct(V2)
#Get the genres
combination_of_genres <- movies_table %>% distinct(V3)
concatenated <- paste(combination_of_genres$V3, collapse = '|')
individual_genres_ocurrence <- strsplit(concatenated, "[|]")
genres <- levels(factor(unlist(individual_genres_ocurrence))) #18 Gêneros nesse dataset

genre_names <- as.matrix(genres)
genre_names <- paste0("",as.character(genre_names),sep="")

movies_processed <-
  matrix(
    0L,
    nrow = nrow(movies_table),
	ncol = length(genre_names)
  )

movies_id <- paste("movie",as.character(movies_id),sep="")
  
colnames(movies_processed) <- c(genre_names)
rownames(movies_processed) <- movies_id


max_length <- nrow(movies_table)
iteration_index = 0
for(i in 1:max_length){
  iteration_index = iteration_index + 1  
  movie_genres = unlist(strsplit(toString(movies_table[i,"V3"]), "[|]"))

  if(length(movie_genres) > 0){
	for(j in 1:length(movie_genres)){
		movies_processed[i,movie_genres[[j]]] <- 1
	}
  }
}

#Users
#Users Table Structure: UserID::Gender::Age::Occupation::Zip-code
users <- users_table %>% distinct(V1) #1
genders <- users_table %>% distinct(V2) #1
ages <- users_table %>% distinct(V3) #7
occupations <- users_table %>% distinct(V4) #7
zips <- users_table %>% distinct(V5) #7

users_id <- as.matrix(users)
users_id <- as.matrix(paste("user",as.character(users_id),sep=""))

gender_names <- as.matrix(genders)
gender_names <- paste0("genders",as.character(gender_names),sep="")

age_names <- as.matrix(ages)
age_names <- paste0("age",as.character(age_names),sep="")

occupation_names <- as.matrix(occupations)
occupation_names <- paste0("job",as.character(occupation_names),sep="")

users_colnames <- c(gender_names,age_names,occupation_names)

users_processed <-
  matrix(
    0L,
    nrow = nrow(users_id),    
	ncol = length(users_colnames)
  )

colnames(users_processed) <- users_colnames
rownames(users_processed) <- users_id

max_length_u <- nrow(users_processed)
iteration_index_u = 0
for(i in 1:max_length_u){
	iteration_index_u = iteration_index_u + 1        
	columns_u = c(paste("genders",toString(users_table[i,"V2"]),sep=""),paste("age",toString(users_table[i,"V3"]),sep=""),paste("job",toString(users_table[i,"V4"]),sep=""))
	users_processed[i,columns_u] = 1
}

#MOVIE RATING PROCESSING
#Ratings Table Structure:UserID::MovieID::Rating::Timestamp
data_columns = c(users_id,users_colnames,movies_id,genre_names,"Timestamp","Rating")	
parts = 1000 #Breaks Movilens 1M to 1000 files at first
max_length_r = as.integer(nrow(ratings_table)/parts)
gc()
start_time <- Sys.time()
for(i in 1:parts){	
	data_processed = matrix(0L,nrow = max_length_r,ncol = (length(data_columns)))
	colnames(data_processed) <- data_columns
	init = (i-1)*max_length_r
	
	iteration_index_r = 0
	for(j in 1:max_length_r){
		iteration_index_r = iteration_index_r + 1
		actual_user = paste("user",ratings_table[j+init,"V1"],sep="")
		actual_movie = paste("movie",ratings_table[j+init,"V2"],sep="")	
		
		columns_from_user = names(users_processed[actual_user, unlist(users_processed[actual_user,]) == 1])
		columns_from_movie = names(which(movies_processed[actual_movie,]==1))
		data_processed[j,c(actual_user,actual_movie,columns_from_user,columns_from_movie)] = 1
		data_processed[j,"Timestamp"] = ratings_table[j+init,"V4"]
		data_processed[j,"Rating"] = ratings_table[j+init,"V3"]	
	}		
	write.table(data_processed, paste("data_processed_",i,".csv"), sep=",") 
	rm(data_processed)
	gc()
}

end_time <- Sys.time()
print("Elapsed time:")
print(end_time - start_time)
beep()


