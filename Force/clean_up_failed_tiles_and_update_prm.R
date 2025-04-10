path_to_prm = 'X:/eoagritwin/force/parameterfile/TSA_BRANDENBURG.prm'
path_to_force = 'X:/eoagritwin/force/output/BRANDENBURG/2023/'
number_of_expected_bands = 48

prm_lines = readLines(path_to_prm)

# Optionally, split the values from the labels
xx_values <- strsplit(sub("^%XX%:\\s*", "", lines[1]), " ")[[1]]
yy_values <- strsplit(sub("^%YY%:\\s*", "", lines[2]), " ")[[1]]

# Convert character vectors to numeric (optional)
xx_values <- as.numeric(xx_values)
yy_values <- as.numeric(yy_values)

# make the folder combinations 
folders <- character(length(xx_values))
for(i in 1:length(xx_values)){
  folders[i] <- (paste0('X00',xx_values[i],'_Y00',yy_values[i]))
}


x_del = numeric()
y_del = numeric()
#search, delete folders and find x and y for .prm update
for(i in 1:length(folders)){
  if(length(list.files(paste0(path_to_force, folders[i]), recursive = T)) < number_of_expected_bands){
    unlink(paste0(path_to_force, folders[i]), recursive = TRUE)
    x_del <- c(x_del, as.numeric(sub("X0*(\\d+)_Y0*\\d+", "\\1",  folders[i])))
    y_del <- c(y_del, as.numeric(sub("X0*\\d+_Y0*(\\d+)", "\\1",  folders[i])))
  }
}

# update prm file
new_xx_line <- paste("%XX%:", paste(x_del, collapse = " "))
new_yy_line <- paste("%YY%:", paste(y_del, collapse = " "))

prm_lines[1] <- new_xx_line
prm_lines[2] <- new_yy_line

writeLines(prm_lines, path_to_prm)

