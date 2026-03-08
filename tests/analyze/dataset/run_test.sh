#!/bin/bash

_pwd=$(pwd)
script_path=$(realpath $(dirname $0))
script_relative_path=$(echo $script_path | awk -F'/tests/' '{print $2}')

cd ${script_path}
rm -rf inputs
cp -rL inputs.clean inputs

echo "[do] Running commands in ${script_relative_path} ..."
dock analyze dataset edge inputs -t 0 -j 4
sleep 1
dock analyze dataset split inputs -t 0 -j 4
sleep 1
echo "[done] Running commands"

echo "mv dataset_split.json inputs/dataset_split.json"
mv dataset_split.json inputs/dataset_split.json

echo "[do] Checking ..."
for f in $(ls inputs.bak); do
  bash ../../check_file.sh $f inputs/$f inputs.bak/$f
done
echo "[done] Checking"
cd ${_pwd}
