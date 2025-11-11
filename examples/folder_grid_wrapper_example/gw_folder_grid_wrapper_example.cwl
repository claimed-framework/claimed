cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: local/claimed-gw-folder-grid-wrapper-example:0.1
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  sgw_source_folder:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --sgw_source_folder
  sgw_target_folder:
    type: string
    default: "sgw_source_folder"
    inputBinding:
      position: 4
      prefix: --sgw_target_folder


outputs: []
