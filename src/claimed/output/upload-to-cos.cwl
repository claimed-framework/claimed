cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: romeokienzler/claimed-upload-to-cos:0.8
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  target:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --target
  source_file_pattern:
    type: string
    default: None
    inputBinding:
      position: 4
      prefix: --source_file_pattern
  find_recursive:
    type: bool
    default: True
    inputBinding:
      position: 5
      prefix: --find_recursive
  process_target_file_pattern:
    type: string
    default: None
    inputBinding:
      position: 6
      prefix: --process_target_file_pattern


outputs: []
