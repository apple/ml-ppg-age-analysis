
## Documentation

This repo provides an example pipeline to (i) fit a PPG model with SSL, (ii) fit a
simple ridge regression-based age model, and (iii) analyze age predictions
to re-create statistics + visualizations in the paper (on synthetically generated data).


## Overview


### Setup

Create new environment, (~5 min)
```
conda create --name ppg-age python==3.11
conda activate ppg-age
pip install -r requirements.txt
```


### Training a Ppg Model

Configuration-based and modular pipeline for pre-training (~15 min)
```
python -m training.main -c TrainingPpgSsl
```

This code trains the model on dummy randomly generated PPG data and store artifacts
in the ./artifacts folder. The training details and code adapted from the
[PPG foundation model work](https://arxiv.org/pdf/2312.05409). To train on your own
data, replace `MinimalDataset` with a sample-style Pytorch `torch.data.utils.Dataset`
class.


### Train Age Model on Ppg Embeddings

Run a simple regression model on the output of the Ppg model fitting
```
python -m analysis.run_fit_age_model --embeddings ./artifacts/test_inferences.csv
```
which outputs age predictions for each row in the `test_inferences.csv` dataset
in file `./artifacts/age-model-data/<example-files>.csv`.


### Run Analysis of Age Predictions

Create plotting data and a subset of analysis plots (~5 min)

```
python -m analysis.run_analysis --artifact-dir ./artifacts
```

This creates output data and plots in the output dir as defined in the script `OUTPUT_DIR = ./output-figs`
and populates it with sub directories for each figure.

* Create relative risk plots (ggplot) (~1 min)
```
Rscript make_r_figs.R
```


### System Info

Scripts tested with environment: 

```
%> conda --version
conda 24.11.3

(ppg-age) %> python --version
Python 3.11.0

(ppg-age) %> pip freeze
asttokens==3.0.0
autograd==1.8.0
autograd-gamma==0.5.0
contourpy==1.3.3
cycler==0.12.1
decorator==5.2.1
executing==2.2.0
filelock==3.18.0
fonttools==4.59.0
formulaic==1.2.0
fsspec==2025.7.0
interface-meta==1.3.0
ipython==9.4.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.1
kiwisolver==1.4.8
lifelines==0.30.0
lightning-utilities==0.15.2
MarkupSafe==3.0.2
matplotlib==3.10.5
matplotlib-inline==0.1.7
mpmath==1.3.0
narwhals==2.0.1
networkx==3.5
numpy==2.3.2
packaging==25.0
pandas==2.3.1
parso==0.8.4
patsy==1.0.1
pexpect==4.9.0
pillow==11.3.0
prompt_toolkit==3.0.51
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.19.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
six==1.17.0
stack-data==0.6.3
statsmodels==0.14.5
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.8.0
torchaudio==2.8.0
torchmetrics==1.8.1
torchvision==0.23.0
traitlets==5.14.3
typing_extensions==4.14.1
tzdata==2025.2
wcwidth==0.2.13
wrapt==1.17.2
```
