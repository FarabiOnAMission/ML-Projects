# SST-2 Sentiment Classification

Develop a reproducible binary sentiment classifier for SST-2 movie-review sentences.
The planned workflow is to explore the data, build a simple baseline, fine-tune a
Transformer on Kaggle GPUs, and compare validation accuracy and error patterns.
No datasets, pretrained models, or training runs are included in this setup.

## Existing local environment

Local development uses macOS, Python **3.14.6**, and the existing `.venv`.
Do not recreate this environment or change its Python version. From the project root:

```sh
source .venv/bin/activate
python test_environment.py
python -m pip check
```

The environment smoke test reports versions and accelerator availability; it does
not fail when a GPU is unavailable. The setup check passed all six imports and
`pip check`. This agent's sandbox reported `MPS built: True`, `MPS available: False`;
the owner previously verified MPS successfully. Recheck in a normal local terminal
before GPU work; no environment changes were made to address that discrepancy.

`requirements.txt` lists direct scientific, NLP, notebook, configuration, and
Kaggle dependencies already installed, rather than every transitive dependency.
Most constraints retain the observed minor release and allow patch updates.
The macOS PyTorch constraint follows the installed release; Linux allows
`torch>=2.5,<3` so a compatible Kaggle CUDA build can be retained. This Linux range
has not been tested on Kaggle. Dependencies require Python **3.12 or newer** at
this baseline; the existing local Python remains unchanged.

Installation is only needed in a **separate fresh environment** or if dependencies
are actually missing, using that environment's Python:

```sh
python -m pip install -r requirements.txt
python -m pip check
```

Do not run this install just to repeat setup in the working local `.venv`.
This requirements file is a compatibility specification, not a cross-platform
lockfile. Record the resolved versions for each future experiment.

## VS Code and notebooks

Open the **project root folder** in VS Code. Workspace settings select
`${workspaceFolder}/.venv/bin/python` and activate it in new terminals. The
recommended extensions are Python and Jupyter; global VS Code settings are untouched.
If VS Code has cached another interpreter, run **Python: Select Interpreter** and
choose `.venv/bin/python`. In the notebook kernel picker, select
**Python Environments → .venv/bin/python** (or **SST-2 (.venv)** when listed).
The existing notebook is preserved exactly, including its `python3` kernel metadata.

To launch JupyterLab with that kernel reliably from the project root:

```sh
JUPYTER_PATH="$PWD/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}" .venv/bin/python -m jupyterlab
```

Alternatively use **Terminal → Run Task → SST-2: Open JupyterLab (.venv)** in
VS Code. The project-local `python3` kernelspec points explicitly to the existing
virtual environment; it is ignored by Git because it contains an absolute local
path. No system/user kernels or `.venv` files are modified. After moving the
project, update that local kernelspec path or select the environment directly.
On a fresh checkout, select the new environment directly in VS Code; for
standalone Jupyter, register its kernel under the checkout with:

```sh
python -m ipykernel install --prefix "$PWD/.jupyter" --name python3 --display-name "SST-2 (.venv)"
```

The `--prefix` command writes under `.jupyter/share/jupyter/kernels`; use
`JUPYTER_PATH="$PWD/.jupyter/share/jupyter" python -m jupyterlab` for that registration.
The current checkout uses `.jupyter/kernels` and the earlier launch command.

Run the notebook's first cell to verify `sys.executable` ends in this project's
`.venv/bin/python`. Setup verified the existing notebook by executing it in memory
through this kernel; the saved notebook remained byte-for-byte unchanged. Imports
reported an optional `IProgress`/`ipywidgets` warning; widget dependencies were not
added to the working environment. A sandbox process-inspection restriction also
produced a kernel-shutdown warning after successful execution.
Keep reusable logic in `src/`, and use notebooks for exploration.
Launch tools from the project root. In notebooks, add the repository root to
`sys.path` if importing `src` from the `notebooks/` directory; there is no installed
Python distribution or placeholder training implementation yet.

## Directory layout

```text
SST2-Sentiment/
├── .git/                       # local repository metadata
├── .gitignore
├── .jupyter/kernels/python3/kernel.json  # ignored local kernel
├── .venv/                      # existing environment, ignored
├── .vscode/
│   ├── extensions.json
│   ├── settings.json
│   └── tasks.json
├── README.md
├── requirements.txt
├── configs/.gitkeep
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── notebooks/01_data_exploration.ipynb
├── outputs/
│   ├── models/.gitkeep
│   ├── checkpoints/.gitkeep
│   └── predictions/.gitkeep
├── src/__init__.py
├── tests/__init__.py
└── test_environment.py
```

Empty `.gitkeep` files preserve non-package directories in Git. `configs/`,
notebooks, and artifact folders are not Python packages. `tests/` is ready for
meaningful tests as reusable code is implemented; there are no invented test stubs.

## Development workflow

1. Activate `.venv`, verify the interpreter, and run the environment checks above.
2. Continue exploration in the existing notebook. Download data only when starting
   that work explicitly; put original data in `data/raw/` and derived data in
   `data/processed/`. Keep split boundaries intact to avoid validation leakage.
3. Add reusable preprocessing, evaluation, and later training code in `src/`.
   Store non-secret experiment settings in `configs/`; include seeds, model name,
   dataset version, hyperparameters, and output paths when training is implemented.
4. Add behavioral tests in `tests/`. The existing `test_environment.py` remains at
   the root and should be run directly; no new test runner was installed.
5. Save weights, checkpoints, and predictions in their respective `outputs/`
   subfolders. Keep secrets in environment variables or ignored local credential
   files. Git ignore rules protect common secret filenames, not arbitrary inline
   secrets; inspect file contents and notebook outputs before committing.
6. Inspect `git status --short` and `git diff` before staging. Git is local only;
   no initial commit, remote, upload, or push was performed by this setup.

## Future Kaggle GPU workflow

This is a plan for later work; setup has not connected to Kaggle or started training.

1. Start a Kaggle Linux notebook with a GPU accelerator. Inspect its Python version
   and existing package versions before installation. Do not copy `.venv` from
   macOS. If Kaggle's Python is older than 3.12, this dependency baseline cannot
   be installed unchanged: choose and validate an older compatible dependency
   profile for that runtime rather than changing the local Python installation.
2. Verify `torch.cuda.is_available()` and inspect `torch.__version__`,
   `torch.version.cuda`, and the GPU driver. The requirements deliberately omit
   explicit CUDA packages and wheel URLs: the correct build depends on Kaggle's
   runtime. Retain a preinstalled CUDA-enabled PyTorch within `>=2.5,<3`.
   If incompatible, select a build supported by that runtime before training;
   do not blindly install the local macOS PyTorch version or use `--upgrade`.
3. Transfer source, configs, requirements, and the intended notebook through your
   chosen workflow. Preview dependency resolution with
   `python -m pip install --dry-run -r requirements.txt` (requires network access)
   and review any proposed changes to PyTorch/CUDA or Kaggle's existing libraries.
   Install only when needed, restart the kernel after package changes, and run
   `python -m pip check` and `python test_environment.py` in the repository root.
   A shared Kaggle environment can have unrelated conflicts; resolve those in
   that remote runtime before starting training.
4. Use Kaggle's own notebook Python kernel, not the ignored macOS kernelspec.
   Load future credentials through Kaggle Secrets or environment variables.
   Attach datasets/models deliberately when ready; `/kaggle/input` is for inputs
   and `/kaggle/working` for generated files. Configure output paths accordingly.
5. After preprocessing and training code exists, first perform a small smoke run,
   then train with a fixed seed and explicit settings. Enable mixed precision only
   after checking GPU support. Save checkpoints/predictions under the working
   directory and record the resolved dependency versions and source revision.
6. Evaluate on the labeled validation split and report accuracy plus error analysis.
   Check the dataset's label availability before treating the official test split
   as an evaluation set. Keep model artifacts out of Git and export them separately.

## Git exclusions

`.gitignore` excludes `.venv`, Python/tool caches, Jupyter checkpoints, downloaded
data, model files, experiment logs, temporary files, credentials, and generated
outputs. Only directory markers under `data/` and `outputs/` are retained.
Source notebooks remain versionable; clear sensitive or bulky outputs before a
future commit. The local Jupyter kernelspec is intentionally untracked.
