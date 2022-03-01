# `mle-hyperopt` Examples

- Learn about the search strategy API: `jupyter notebook getting_started.ipynb` or [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mle-infrastructure/mle-hyperopt/blob/main/examples/getting_started.ipynb)

- Getting started with iterative search strategies on a toy MLP and with job scheduling via the `mle-scheduler`: `cd iter_search`
    - Successive Halving: `python run_sh_mlp.py`
    - Hyperband: `python run_hb_mlp.py`
    - PBT: `python run_sh_mlp.py`

- Using the `mle-search` command line interface: `cd mle_search`:

```
mle-search run_mle_search.py -base base.yaml -search search.yaml -iters 10
```