# Oxford-IIT Pet

Notebook-first computer vision project using the **Oxford-IIIT Pet** dataset.
The folder name follows the requested `CV/Oxford-IIT Pet` naming.

## Status and scope

Initial environment and workspace setup only. No dataset, pretrained weights,
training runs, metrics or model artifacts have been downloaded or produced.
Task selection is pending: the dataset supports breed classification and pixel
segmentation. Input is a pet image; the output and evaluation metrics will be
chosen during exploration. Classification would use accuracy and macro F1;
segmentation would use IoU/Dice. These are planning options, not results.

Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
Torchvision loader: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
Review source usage terms and annotations before downloading.

## Structure

```text
Oxford-IIT Pet/
├── .venv/                         # Isolated local Python environment
├── .vscode/                       # Interpreter and notebook support
├── .jupyter/                      # Local kernel, generated and ignored
├── configs/experiment_config.yaml # Initial paths and seed; task undecided
├── data/{raw,processed}/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_inference.ipynb
│   └── 04_error_analysis.ipynb
├── kaggle/README.md               # Future remote execution workspace
├── src/                           # Future reusable code
├── outputs/{models,checkpoints,metrics,figures,predictions}/
├── requirements.txt               # Pinned direct dependencies
├── requirements-macos.lock.txt    # Full local dependency snapshot
└── environment.json               # Verified local environment details
```

## Local use

Open this folder in VS Code, then use its terminal:

```bash
cd "/Users/farabi/Farabi/ML-Projects/CV/Oxford-IIT Pet"
source .venv/bin/activate
export JUPYTER_PATH="$PWD/.jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
python -m jupyterlab
```

In a VS Code notebook, use **Select Kernel → Python Environments → .venv/bin/python**,
or choose **Python (Oxford-IIT Pet)** if listed. The local kernel is stored inside
this project; no global environment or global kernel registration is needed.

The setup uses the installed Python 3.14.6 after verifying native binary wheels.
To recreate it on a compatible Apple Silicon Mac with Python 3.14:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos.lock.txt
python -m ipykernel install --prefix "$PWD/.jupyter" --name oxford-iit-pet --display-name "Python (Oxford-IIT Pet)"
```

The lockfile describes macOS only. Create a fresh Linux environment for Kaggle
and check its installed CUDA/PyTorch versions before changing dependencies.
Never copy `.venv` between operating systems. Run `deactivate` to leave the venv.

## Dependencies

PyTorch + torchvision provide models, datasets and image transforms; timm adds
vision model options. Hugging Face `datasets` provides `load_dataset` for dataset loading. Pillow handles images; NumPy, pandas and scikit-learn support
analysis and evaluation; Matplotlib and seaborn provide plots. JupyterLab,
ipykernel and ipywidgets support notebooks. tqdm, PyYAML and python-dotenv cover
progress/configuration; Kaggle CLI is installed for later remote work.

## Next steps

1. Explore the dataset and choose classification or segmentation.
2. Establish validation and test handling, then a simple baseline.
3. Develop training interactively, saving models and preprocessing together.
4. Load saved artifacts for inference and error analysis.

This project belongs to the parent `ML-Projects` repository. Data, environments,
credentials and large model artifacts are excluded. No nested Git repository,
commit or push is part of initialization.

## Initialization verification

Package consistency, core imports, torchvision image transforms and NMS, notebook
format/syntax, and the exploration notebook setup cell passed. Jupyter discovers
the project kernel with the documented `JUPYTER_PATH`. PyTorch was built with MPS
support, but MPS was unavailable to the setup process; CPU was selected. Recheck
GPU availability in VS Code on the host before choosing training hardware.
Dataset loading and training remain unverified because this step initializes only
the environment and structure. See `environment.json` for installed versions.
