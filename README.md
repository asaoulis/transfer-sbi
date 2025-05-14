# transfer-sbi

Transfer learning for simulation-based inference on the [CAMELS Multifield Dataset](https://camels-multifield-dataset.readthedocs.io/en/latest/index.html). 

## Usage

### Installation

### Running the code

The two main scripts are `train.py` and `eval.py`. Training is for a given configuration file:

```
python train.py <experiment name>
```

where the experiment name should be defined in `config/experiments.py`. The default configuration is shown in `config/default.py`. 

Then, once some models are trained, you can run evaluation over experiments (and sub-tasks) using

```
python eval.py
```

Here, I've opted to just select the experiments in `eval.py`. This runs the entire evaluation suite across all matched runs for a given experiment. 