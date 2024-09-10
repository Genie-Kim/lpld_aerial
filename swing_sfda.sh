#!/bin/bash

# Define the directory containing the YAML files
CONFIG_DIR="configs/sfda"
CONFIG_NAME="sfda_dota2uavdt"

# copy to the parent directory
cp "$CONFIG_DIR/$CONFIG_NAME.yaml" "configs/sfda/exp/"

# Define the configurations to update
TMUXNAME="tm014"
NEW_METHOD="LPLD"
NEW_BASE_LR="0.001"
NEW_EMAPERIOD="2000"
NEW_KEEP_RATE="0.70"
GPUNUM="2"

# split NEW_BASE_LR with '.' and get the last element
IFS='.' read -r -a array <<< "$NEW_BASE_LR"
TEMP_BASE_LR="${array[1]}"

# split NEW_KEEP_RATE with '.' and get the last element
IFS='.' read -r -a array <<< "$NEW_KEEP_RATE"
TEMP_KEEP_RATE="${array[1]}"




# make a array with the above new configuration and join them with "_"
NEW_CONFIG_NAME=""$CONFIG_NAME"_"$NEW_METHOD"_"$TEMP_BASE_LR"_"$NEW_EMAPERIOD"_"$TEMP_KEEP_RATE""


file="configs/sfda/exp/$NEW_CONFIG_NAME.yaml"

mv "configs/sfda/exp/$CONFIG_NAME.yaml" "$file"

# Loop through all YAML files in the directory
echo "Updating $file..."

# Update the configurations using yq
yq eval ".SOURCE_FREE.METHOD = \"$NEW_METHOD\"" -i "$file"
yq eval ".SOLVER.BASE_LR = $NEW_BASE_LR" -i "$file"
yq eval ".SOURCE_FREE.EMAPERIOD = $NEW_EMAPERIOD" -i "$file"
yq eval ".SOURCE_FREE.KEEP_RATE = $NEW_KEEP_RATE" -i "$file"


echo "Updated $file"

# generate new tmux session
tmux new-session -d -s "$TMUXNAME"
sleep 5
tmux send-keys -t "$TMUXNAME" 'zsh' ENTER
sleep 5
tmux send-keys -t "$TMUXNAME" 'lpld' ENTER
sleep 5
tmux send-keys -t "$TMUXNAME" "CUDA_VISIBLE_DEVICES=$GPUNUM python tools/train_main.py --config-file $file --model-dir source_model/dota_source_v11/best_mAP.pth" ENTER