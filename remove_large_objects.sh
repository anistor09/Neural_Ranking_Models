#!/bin/bash


LARGE_OBJECTS=$(cat packfile_contents.txt | sort -k 3 -n | tail -20 | awk '{print $1}')

# Create a temporary file to hold the large object IDs
TEMP_FILE=$(mktemp)
for object in $LARGE_OBJECTS; do
    echo $object >> $TEMP_FILE
done

# Remove the large objects using git-filter-repo
git filter-repo --strip-blobs-with-ids $TEMP_FILE --force

