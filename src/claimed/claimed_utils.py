# Utilities used by CLAIMED

import os
import re
import sys
import zipfile

# compresses 'path' into 'target' zipfile


def zipdir(target, path):
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(root, file))

# uncompresses 'zipfile_name' to 'target' directory


def unzip(target, zipfile_name):
    with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
        zip_ref.extractall(target)

def parse_args_to_parameters():
  # override parameters received from a potential call using %run magic
  parameters = list(
      map(
          lambda s: re.sub('$', '"', s),
          map(
              lambda s: s.replace('=', '="'),
              filter(
                  lambda s: s.find('=') > -1 and s.find('[A-Za-z0-9_]*=[.\/A-Za-z0-9]*') > -1,
                  sys.argv
              )
          )
      )
  )