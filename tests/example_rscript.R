# Reading env variables

name <- Sys.getenv('name')

default <- Sys.getenv('default', "default")

number <- as.numeric(Sys.getenv('number', 10))

print(paste("hello", name))

print(number)

# apt install libgdal-dev

# Install packages
install.packages('readr')
library(readr)
