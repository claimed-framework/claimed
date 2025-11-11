#!/usr/bin/env cwl-runner


cwlVersion: v1.2

# What type of CWL process we have in this document.
#class: CommandLineTool

class: Workflow

inputs:
  num_values: string


outputs: []

steps:
  example1:
    run: operator_example.cwl
    in: 
      num_values: num_values
    out: []

  example2:
    run: operator_example.cwl
    in: 
      num_values: num_values
    out: []

