cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: docker.io/mdorzweiler/claimed-input-xview-download:0.1
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  username:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --username
  password:
    type: string
    default: None
    inputBinding:
      position: 4
      prefix: --password
  move_to_dir:
    type: string
    default: None
    inputBinding:
      position: 5
      prefix: --move_to_dir
  chromedriver_path:
    type: string
    default: None
    inputBinding:
      position: 6
      prefix: --chromedriver_path
  max_download_time:
    type: string
    default: None
    inputBinding:
      position: 7
      prefix: --max_download_time
  label:
    type: string
    default: None
    inputBinding:
      position: 8
      prefix: --label


outputs: []
