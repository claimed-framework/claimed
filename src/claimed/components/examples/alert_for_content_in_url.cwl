cwlVersion: v1.2
class: CommandLineTool

baseCommand: "claimed"

inputs:
  component:
    type: string
    default: romeokienzler/claimed-alert-for-content-in-url:0.6
    inputBinding:
      position: 1
      prefix: --component
  log_level:
    type: string
    default: "INFO"
    inputBinding:
      position: 2
      prefix: --log_level
  url_to_notify:
    type: string
    default: None
    inputBinding:
      position: 3
      prefix: --url_to_notify
  url_to_query:
    type: string
    default: None
    inputBinding:
      position: 4
      prefix: --url_to_query
  filter_content:
    type: string
    default: None
    inputBinding:
      position: 5
      prefix: --filter_content
  sleep:
    type: int
    default: 30
    inputBinding:
      position: 6
      prefix: --sleep


outputs: []
