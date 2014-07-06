# Extreme ANFIS

Extreme learning in adaptive fuzzy inference system

### Structure

- The functions are split in packages in `packages` directory.
- Each directory has a training function and error functions (`rmse.m` and `error.m`).
- Each package has separate `README.md` describing the working of functions.
- `rmse.m` is for calculating root mean squared error.
- `error.m` finds percent error for classification (binary) task.
- Directory `tests` contains test code organised according to package names.
- Directory `data` contains datasets for classification tasks.
- `docs` contains few docs including the paper on **Extreme ANFIS**.

### Current state

- Simple **Extreme ANFIS** is made as proposed in paper.
  - *Performs better than ANFIS as far as speed is considered, accuracy also okay.*
- Ensemble of **Extreme ANFIS** is made using bagging.
  - *Works. But needs tweaking. Seems better to boost rather than bag.*
- Genetic **Extreme ANFIS**. The random input parameters are evolved.
  - *No significant improvement. Good chances of wrong implementation.*
- Additive ensemble of **Extreme ANFIS**.
  - *Stellar !! Faster and more accurate !*

### Testing data
Head over [here](https://archive.ics.uci.edu/ml/datasets.html).
There are nice benchmark datasets there.
Also please keep upating this doc for anything that we can do here.
