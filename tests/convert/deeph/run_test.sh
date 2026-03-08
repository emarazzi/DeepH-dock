#!/bin/bash

_pwd=$(pwd)
script_path=$(realpath $(dirname $0))
script_relative_path=$(echo $script_path | awk -F'/tests/' '{print $2}')

cd ${script_path}
rm -rf legacy updated standardize minus_core

echo "[do] Running commands in ${script_relative_path} ..."
dock convert deeph upgrade legacy.bak updated -t 0 -j 2
sleep 1
dock convert deeph downgrade updated.bak legacy -t 0 -j 2
sleep 1
cp -r updated standardize
dock convert deeph standardize standardize --overwrite -t 0 -j 2
sleep 1
for d1 in $(ls updated); do
  dock convert deeph minus-core updated/$d1 minus_core/$d1 single_atoms.bak -t -1 -j 1
done
sleep 1
echo "[done] Running commands"

echo "[do] Checking ..."
for d1 in $(ls updated); do
  for f in $(ls updated/$d1); do
    bash ../../check_file.sh $f updated/$d1/$f updated.bak/$d1/$f
  done
done
for d1 in $(ls legacy); do
  for f in $(ls legacy/$d1); do
    bash ../../check_file.sh $f legacy/$d1/$f legacy.bak/$d1/$f
  done
done
for d1 in $(ls standardize); do
  for f in $(ls standardize/$d1); do
    bash ../../check_file.sh $f standardize/$d1/$f standardize.bak/$d1/$f
  done
done
for d1 in $(ls minus_core); do
  for f in $(ls minus_core/$d1); do
    bash ../../check_file.sh $f minus_core/$d1/$f minus_core.bak/$d1/$f
  done
done
echo "[done] Checking"
cd ${_pwd}
