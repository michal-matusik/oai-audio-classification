"""Download official datasets linked in the task notebook."""

from pathlib import Path

import gdown

FILES = {"train.npz": "1WpPqcKTh_jzCsBm7k32IRtqlsxgpV8u4", "val.npz": "1PwVhwFAGX3cS9gA7SYbJORjpQroPVQEN"}


if __name__ == "__main__":
    output = Path("data")
    output.mkdir(exist_ok=True)
    for name, file_id in FILES.items():
        gdown.download(id=file_id, output=str(output / name), quiet=False)
