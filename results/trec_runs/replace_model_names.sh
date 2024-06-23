#!/bin/bash

# Set the directory to the directory where the script is located
directory=$(dirname "$0")

# Loop through each file in the specified directory
for file in "$directory"/*; do
    # Extract the filename from the path
    filename=$(basename "$file")

    # Extract substring from filename after the first underscore
    substring=${filename#*_}

    # Replace all occurrences of 'model_names.sh' with the extracted substring in the file
    sed -i "s/model_names.sh/$substring/g" "$file"
done

echo "Replacement complete."
