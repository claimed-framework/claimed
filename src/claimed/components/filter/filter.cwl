cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: romeokienzler/claimed-filter:0.6
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  predicate:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --predicate
  file_name:
    type: string
    default: None
    inputBinding:
      position: 4
      prefix: --file_name
  output_file_name:
    type: string
    default: None
    inputBinding:
      position: 5
      prefix: --output_file_name


outputs: []
