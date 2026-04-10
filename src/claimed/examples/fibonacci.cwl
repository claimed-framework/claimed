cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: docker.io/mdorzweiler/claimed-fibonacci:0.1
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  b:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --b


outputs: []
