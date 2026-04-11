#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["/opt/app-root/bin/ipython","/opt/app-root/src/filter.py"]
hints:
  DockerRequirement:
    dockerPull:  romeokienzler/claimed-filter:0.5
inputs:
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 1
      prefix: --log_level
  predicate:
    type: string
    default: None
    inputBinding:
      position: 2
      prefix: --predicate
  file_name:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --file_name
  output_file_name:
    type: string
    default: None
    inputBinding:
      position: 4
      prefix: --output_file_name

outputs:
  dummy_out:
    type: File
    outputBinding:
      glob: query_result.csv