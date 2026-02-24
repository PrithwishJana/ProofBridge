#!/bin/bash

#--------------Ensure I'm at correct path----------------
if [ "$(basename "$PWD")" != "PartitionAndProve" ]; then
  echo "❌ Please run this script from inside ./PartitionAndProve"
  exit 1
fi
echo "✅ You are in ./PartitionAndProve"
#----------------

#--------------Activate virtual environment----------------
module load gcc arrow/15.0.1 opencv/4.11.0
source ~/lean_env/bin/activate
#----------------

#--------------Creates a tmp folder for REPL----------------
python ./LEAN_interaction/checkLEAN.py 
#(Takes 5 minutes almost)
#----------------

#--------------Moves tmp folder to appropriate place----------------
# ensure destination dir exists
if [ ! -d "../autoformalization-jtemb" ]; then
  mkdir ../autoformalization-jtemb
  echo "✅ Created ../autoformalization-jtemb"
fi

# now move tmpFolder if it's not already there
if [ ! -d "../autoformalization-jtemb/tmpFolder" ]; then
  mv ./tmpFolder ../autoformalization-jtemb/
  echo "✅ Moved tmpFolder into ../autoformalization-jtemb/"
else
  echo "ℹ️ tmpFolder already exists inside ../autoformalization-jtemb"
fi
#----------------