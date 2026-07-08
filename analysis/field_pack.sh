#!/bin/bash

RUNDIR=$(pwd)/../output
mkdir -p ${RUNDIR}
cd ${RUNDIR}



PARAMS=$(cat <<EOF
{
  "datafolder": "ma40",
  "njobs": 8,
  "xmin": 1200,
  "xlen": 250,
  "ymin": -204.8,
  "ylen": 409.6,
  "tmin": 0,
  "tlen": 100000,
  "keylist": ["fN_1", "favgB1","favgB2","favgB3",  "favgE1","favgE2","favgE3" ]
}
EOF
)

python3 ../analysis/field_pack.py --params "$PARAMS"
