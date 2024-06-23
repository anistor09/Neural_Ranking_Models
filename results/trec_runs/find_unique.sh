#!/bin/bash

# Assuming the filename is given as the first command line argument
filename="$1"

# Specify the directory path relative to the script location
# Remove the './' if you want to strictly use the same directory or adjust as needed
directory_path=$(dirname "$0")/

# Full path construction
file_path="${directory_path}${filename}"

# Use awk to extract the third column and sort and get unique values
awk '{print $1}' "$file_path" | sort -u | wc -l
