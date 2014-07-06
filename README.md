# Extreme ANFIS

Extreme learning in adaptive fuzzy inference system

### Current state

**The older code is deprecated, still if needed, switch to `custom-fis` branch**

There is a function `exanfis` that takes following inputs
-   `data_train` : Observations arranged in rows, each column is an input variable, except the last one, which is the output.
-   `n_mfs` : An integer specifying the number of membership functions to use for each data variable.
-   `epochs` : The number of times, random guesses should be made.
-   `data_test` : Data for testing purpose.

### Working
-   Current code works by randomly generating input parameters, and checking the one which fits best.
-   Also the case with uniform parameters is considered.

### Problems and Todos
-   Add genetic algorithms.
-   Testing the code. Few special testing cases follows :
    - *Multi output dataset*

### Test notes
-   There are test files like `single_var.m`.
-   Open them in matlab and push `run and time`.
eaemdemacs-   This will display results from profiler, telling everything we need to know about timings.

### Testing data
Head over [here](https://archive.ics.uci.edu/ml/datasets.html).
There are nice benchmark datasets there.
Also please keep upating this doc for anything that we can do here.

### Experiments
-   One thing that we can do is reducing the number of rules.
    - Currently if there are 5 membership functions and 10 attributes, then rules are all that are possible, i.e. 5<sup>10</sup>.
    - But there are techniques that use clustering to reduce the rules considerably based on structure in data.
    - This will ultimately reduce our computation and thus seems a nice thing to try.