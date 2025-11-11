
args = commandArgs(trailingOnly=TRUE)

for (parameter in args) {
  key_value <- unlist(strsplit(parameter, split="="))
  if (length(key_value) == 2) {
    print(parameter)
    key <- key_value[1]
    value <- key_value[2]
    eval(parse(text=paste0('Sys.setenv(',key,'="',value,'")')))
    } else {
    print(paste('Could not find key value pair for argument ', parameter))
    }
}
