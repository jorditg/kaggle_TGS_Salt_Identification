#!/bin/bash

IFS=$'\n'

for file in $(find $1 -name "*.jpg"); do
    python predict-client-highres.py $file
done
