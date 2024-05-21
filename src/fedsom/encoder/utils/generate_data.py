import random
from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_mixed_dataset(num_rows: int, num_sections: int = 5, section_width: int = 5) -> pd.DataFrame:
    cols = ["col_" + str(i + 1) for i in range(20)]
    data_dict = {}  # type: Dict[str,Any]
    for i in range(num_sections):
        data_dict[cols[i]] = [random.choice(["apple", "banana", "cherry", "date"]) for _ in range(num_rows)]
        data_dict[cols[section_width + i]] = np.random.randint(1, 100, size=num_rows)
        data_dict[cols[2 * section_width + i]] = [bool(random.getrandbits(1)) for _ in range(num_rows)]
        data_dict[cols[3 * section_width + i]] = np.random.rand(num_rows)

    return pd.DataFrame(data_dict)
