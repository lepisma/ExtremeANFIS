# Extreme ANFIS

Extreme learning in adaptive fuzzy inference system

### Current state
There is a function `exanfis` that takes following inputs
-   `x_train` : Observations arranged in rows, each column is an attribute.
-   `y_train` : The ouput matrix. Here observations are arranged columnwise (this can be changed to suite better the structure, but thats quite immaterial as of now).
-   `x_test` : Testing data
-   `y_test` : Testing output
-   `n_mfs` : An intger specifying the number of mfs to use for each attribute of data.

This function is expected to return the accuracy on the test data.

*This function is not tested, just blind coded as of now*

### Problems and ToDos
-   The first basic problem is that it uses many `for` loops. Although we dont know yet about how this will affect the speed, but its creepy. We dont expect so many nested `for` loops. So, an attempt should be to **vectorise** the matrix operations happening over here.
-   Code to find the time used in the learning process. This can be better implemented if we put traning code separately and prediction code separately.
-   Testing the code. Few special testing cases follows
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
