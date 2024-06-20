#!/bin/bash

# Set the directory to the directory where the script is located
directory=$(dirname "$0")

# Loop through each .trec file in the specified directory
for file in "$directory"/*.trec; do
    # Extract the filename from the path
    filename=$(basename "$file")

    # Extract substring from filename after the first underscore
    substring=${filename#*_}

    echo "\"$substring\"," >> "model_names.txt"
done

echo "Replacement complete."
