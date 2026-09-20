"""`python -m hotloop_schemas [out_dir]` regenerates the committed JSON Schemas."""

import sys
from pathlib import Path

from hotloop_schemas.io import dump_json_schemas

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "schemas"
    for p in dump_json_schemas(out):
        print(p)
