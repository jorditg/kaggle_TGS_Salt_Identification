#!/bin/bash

IFS=$'\n'

for file in $(ls *.jpeg); do
  name=$(echo $file | cut -b1-16)
  mv $file $name
done
