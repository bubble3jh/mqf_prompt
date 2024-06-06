#!/bin/bash

# Define an array of the bash files to execute sequentially
bash_files=(
  # "train_exp_source_sensors_target_bcg.sh"
  "train_exp_source_sensors_target_ppgbp.sh"
)

# Loop through each bash file and execute it
for file in "${bash_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Executing $file..."
    bash "$file"
  else
    echo "File $file not found!"
  fi
done
