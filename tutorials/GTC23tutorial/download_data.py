import os
import urllib
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import merlin.io
from merlin.core.dispatch import get_lib
from merlin.datasets import BASE_PATH


_FILES = ["ground_truth.csv", "test_set.csv", "train_set.csv"]
_DATA_URL = "https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/"


def download_booking(path: Path):
    """Automatically download the booking dataset.
    Parameters
    ----------
    path (Path): output-path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for file in _FILES:
        local_filename = str(path / file)
        url = os.path.join(_DATA_URL, file)
        desc = f"downloading {os.path.basename(local_filename)}"
        with tqdm(unit="B", unit_scale=True, desc=desc) as progress:

            def report(chunk, chunksize, total):
                if not progress.total:
                    progress.reset(total=total)
                progress.update(chunksize)

            urllib.request.urlretrieve(url, local_filename, reporthook=report)