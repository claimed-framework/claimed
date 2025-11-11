cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: us.ico.io/geodn/claimed-operator-example:0.2
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  input_path:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --input_path
  with_default:
    type: string
    default: "default_value"
    inputBinding:
      position: 4
      prefix: --with_default
  num_values:
    type: string
    default: "5"
    inputBinding:
      position: 5
      prefix: --num_values
  output_path:
    type: string
    default: "None"
    inputBinding:
      position: 6
      prefix: --output_path




outputs: []