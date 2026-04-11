cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: containerless/claimed-operators/train_logistic:0.1.0
    inputBinding:
      position: 1
      prefix: --component
  config:
    type: string
    default: None
    inputBinding:
      position: 2
      prefix: --config


outputs: []
