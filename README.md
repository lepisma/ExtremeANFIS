# Extreme ANFIS

Extreme learning in adaptive fuzzy inference system

### Current state

**The older code is deprecated, still if needed, switch to `custom-fis` branch**

There is a function `exanfis` that takes following inputs
-   `x_train` : Observations arranged in rows, each column is an attribute.
-   `y_train` : The ouput matrix. Here observations are arranged columnwise (this can be changed to suite better the structure, but thats quite immaterial as of now).
-   `n_mfs` : An intger specifying the number of mfs to use for each attribute of data.

This function is expected to return a trained fis that can be used for evaluation.

*This function is not tested, just blind coded as of now*

### Problems and ToDos
-   Generate random mf parameters, rather than uniform.
-   Make an accuracy finding function.
-   Code to find the time used in the learning process.
-   Testing the code. Few special testing cases follows :
    - **Multi output dataset**
    - **Single attribute input dataset**

*Since performance on regression has already been considered, our first aim will be classification. Single class, then multiclass*

### Testing data
Head over [here](https://archive.ics.uci.edu/ml/datasets.html).
There are nice benchmark datasets there.
Also please keep upating this doc for anything that we can do here.

### Experiments
-   One thing that we can do is reducing the number of rules.
    - Currently if there are 5 membership functions and 10 attributes, then rules are all that are possible, i.e. 5<sup>10</sup>.
    - But there are techniques that use clustering to reduce the rules considerably based on structure in data.
    - This will ultimately reduce our computation and thus seems a nice thing to try.
