# Reading env variables

name <- Sys.getenv('name', 'world')

default <- Sys.getenv('default', "default")

number <- as.numeric(Sys.getenv('number', 10))

print(paste("hello", name))

print(number)

# Install packages
install.packages('readr')
library(readr)
