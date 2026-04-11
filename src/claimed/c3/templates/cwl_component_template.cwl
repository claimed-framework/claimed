cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: ${repository}/claimed-${name}:${version}
    inputBinding:
      position: 1
      prefix: --component
${inputs}

outputs: ${outputs}
