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
                         min_J=0,
                         min_K=0,
                         n_bins=3,
                         time_relaxed=True,
                         spm_algo='fournier08closed')

triclustering.fit(data)

triclustering.triclusters_()

```

## Experimental Data 📊🧪

TriHSPAM effectiveness can be assessed with synthetic data with planted triclusters.
Datasets are available in [synthetic datasets folder](/experiments/synthetic_datasets/) together with their settings.

## Citing the Paper 📑

If you use TriHSPAM in your research, please cite our paper:

Soares, D. F., Henriques, R., & Madeira, S. C. (2025). TriHSPAM: Triclustering heterogeneous longitudinal clinical data using sequential patterns. Pattern Recognition, 167, 111762.
https://doi.org/10.1016/j.patcog.2025.111762

```bibtex
@article{soares2025trihspam,
  title={TriHSPAM: Triclustering heterogeneous longitudinal clinical data using sequential patterns},
  author={Soares, Diogo F and Henriques, Rui and Madeira, Sara C},
  journal={Pattern Recognition},
  volume={167},
  pages={111762},
  year={2025},
  publisher={Elsevier}
}
```
