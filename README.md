# GB National PV forecast
Machine learning experiments to forecast national solar PV power in Great Britain.

# Installation

## Installation with conda / mamba
We recommend installing [mamba](https://github.com/mamba-org/mamba).

If installing on a platform without a GPU, then uncomment `- cpuonly` in `environment.yml`.

```shell
mamba env create -f environment.yml
conda activate pv_forecast
pip install -e .
pre-commit install
```

# Names

This git repo is called `gb_national_pv_forecast`. But the Python package is called `pv_forecast`.

# Acknowledgements

Thank you to nvidia for their very generous support: nvidia gave us four RTX A6000 GPUs via the nvidia foundation, and a further two RTX A6000 GPUs via the nvidia hardware grant. Thank you nvidia!