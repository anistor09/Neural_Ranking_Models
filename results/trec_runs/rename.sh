#!/bin/bash

# Set the directory to the directory where the script is located
directory=$(dirname "$0")

# Loop through each file in the specified directory
for file in "$directory"/*; do
    if [[ ! "$file" =~ \.sh$ ]]; then
        mv "$file" "$file.trec"
    fi
done

echo "Rename complete."
