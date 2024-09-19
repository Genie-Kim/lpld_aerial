#!/bin/bash

# Define the directory containing the YAML files
CONFIG_DIR="configs/sfda"
# CONFIG_NAME="sfda_uavdtgtaday2fog"
CONFIG_NAME="sfda_uavdtgtaday2night"

# copy to the parent directory
cp "$CONFIG_DIR/$CONFIG_NAME.yaml" "configs/sfda/exp/"

# Define the configurations to update
NEW_METHOD="MTBASE"
TMUXNAME="tm007"   NEW_BASE_LR="0.0002"   NEW_EMAPERIOD="1"   NEW_KEEP_RATE="0.9996"   GPUNUM="1"

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
tmux send-keys -t "$TMUXNAME" "CUDA_VISIBLE_DEVICES=$GPUNUM python tools/train_main.py --config-file $file --model-dir checkpoint/uavdtgta_source_v1/model_37999.pth" ENTER






# tmux kill-session -t "tm001"
# tmux kill-session -t "tm002"
# tmux kill-session -t "tm003"
# tmux kill-session -t "tm004"
# tmux kill-session -t "tm005"
# tmux kill-session -t "tm006"
# tmux kill-session -t "tm007"
# tmux kill-session -t "tm008"
# tmux kill-session -t "tm009"
# tmux kill-session -t "tm010"
# tmux kill-session -t "tm011"
# tmux kill-session -t "tm012"
# tmux kill-session -t "tm013"
# tmux kill-session -t "tm014"
# tmux kill-session -t "tm015"
# tmux kill-session -t "tm016"
# tmux kill-session -t "tm017"




