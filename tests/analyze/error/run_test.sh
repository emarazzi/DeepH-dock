#!/bin/bash

_pwd=$(pwd)
script_path=$(realpath $(dirname $0))
script_relative_path=$(echo $script_path | awk -F'/tests/' '{print $2}')

cd ${script_path}
rm -rf infer
cp -rL infer.clean infer

echo "[do] Running commands in ${script_relative_path} ..."
dock analyze error entries infer/dft -b benchmark.bak/dft -t 0 -j 4 --cache-res
sleep 1
dock analyze error orbital infer/dft -b benchmark.bak/dft -t 0 -j 4 --cache-res
sleep 1
dock analyze error element-pair infer/dft -b benchmark.bak/dft -t 0 -j 4 --cache-res
sleep 1
dock analyze error element infer/dft -b benchmark.bak/dft -t 0 -j 4 --cache-res --E-range 0.05 0.2
sleep 1
dock analyze error structure infer/dft -b benchmark.bak/dft -t 0 -j 4 --cache-res
sleep 1
echo "[done] Running commands"

echo "[do] Checking ..."
echo "[WARN] The read-in order of data directories are not fixed. Please check the results manually."
echo "[done] Checking"
cd ${_pwd}
