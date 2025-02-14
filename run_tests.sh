#!/bin/bash

std_err_file="~/test-terratorch-iterate.err"

# Using the test command
if test -f "$std_err_file"; then
  echo "File exists."
  rm $std_err_file

std_out_file="~/test-terratorch-iterate.out"

# Using the test command
if test -f "$std_out_file"; then
  echo "File exists."
  rm $std_out_file

jbsub -e ~/test-terratorch-iterate.err -o ~/test-terratorch-iterate.out -m 20G  -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py