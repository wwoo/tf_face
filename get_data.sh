#!/bin/bash

mkdir data

# Download PubFig evaluation dataset
python tf/face_extract/pubfig_get.py tf/face_extract/eval_urls.txt ./data

# We'll have duplicates due to files with the same name. Remove them
cat ./data/manifest.txt | sort | uniq > ./data/manifest_uniq.txt

# Crop faces using the Vision API.  This needs a service account.
python tf/face_extract/crop_faces.py ./data/manifest_uniq.txt $PWD

# Finally split the dataset so we have some validation data
python tf/face_extract/split_data.py ./data/vision-manifest.txt ./data/train.txt ./data/valid.txt
i
# Create a tarball to upload to GCS.
find ./data -type f \( -name "crop_*" -o -name "*.txt" \) -print0 | tar czvf out.tar.gz --null -T -

