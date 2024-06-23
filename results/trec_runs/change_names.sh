#!/bin/bash

# Save the current directory
original_dir=$(pwd)

# Loop through all directories in the current directory
for dir in */; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Change to the directory
        cd "$dir"

        # Loop through all files in this directory
        for file in *; do
            # Check if it's a regular file
            if [ -f "$file" ]; then
                # Extract the filename without the extension
                filename_without_ext="${file%.*}"

                # Replace 'pyterrier' with the filename without the extension in the file
                sed -i "s/pyterrier/${filename_without_ext}/g" "$file"

                echo "Processed $file in directory $dir"
            fi
        done

        # Return to the original directory
        cd "$original_dir"
    fi
done

echo "All directories processed."
