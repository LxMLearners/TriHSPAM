# TriHSPAM: Triclustering Heterogeneous Longitudinal Clinical Data using Sequential Patterns

## How to use 🧐

```python
from TriHSPAM import TriHSPAM
import numpy as np

data = [
    [
        [2, 2, 3],
        [5, 0, 3],
        [9, 3, 5]
                        ],
    [
        ['y', 'x', 'z'],
        ['y', 'y', 'z'],
        ['z', 'z', 'y']
                        ],
    [
       [5, 5, 5],
       [3, 7, 0],
       [1, 2, 0],
                        ]
]

triclustering = TriHSPAM(symb_features_idx=[1],
                         num_features_idx=[0,2],
                         min_I=1,
                         min_J= 0,
                         min_K= 0,
                         n_bins=3,
                         time_relaxed=True,
                         spm_algo='fournier08closed')

triclustering.fit(data)

triclustering.triclusters_()

```

## Citing the Paper 📑

If you use TriHSPAM in your research, please cite our paper:

TBA
