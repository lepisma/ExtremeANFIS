# Extreme ANFIS

Extreme learning in adaptive fuzzy inference system

### Current state

**The older code is deprecated, still if needed, switch to `custom-fis` branch**

There is a function `exanfis` that takes following inputs
-   `x_train` : Observations arranged in rows, each column is an attribute.
-   `y_train` : The ouput matrix. Here observations are arranged columnwise (this can be changed to suite better the structure, but thats quite immaterial as of now).
-   `n_mfs` : An intger specifying the number of mfs to use for each attribute of data.

### Problems and ToDos
-   Generate random mf parameters, rather than uniform.
-   Code to find the time used in the learning process.
-   Testing the code. Few special testing cases follows :
    - **Multi output dataset**
    - **Single attribute input dataset**

*Since performance on regression has already been considered, our first aim will be classification. Single class, then multiclass*

### Test notes
-   There are test files like `curve_test.m`.
-   Open them in matlab and push `run and time`.
-   This will display results from profiler, telling everything we need to know about.
-   There is no code for accuracy as of now, but the graphs in `curve_test.m` are encouraging.
-   But few things are not encouraging
    - `eanfis` is slower for function with single variable.
    - The problem is that most of the time is eaten by `evalfis` in `eanfis`, it can be stripped down.

### Testing data
Head over [here](https://archive.ics.uci.edu/ml/datasets.html).
There are nice benchmark datasets there.
Also please keep upating this doc for anything that we can do here.

### Experiments
-   One thing that we can do is reducing the number of rules.
    - Currently if there are 5 membership functions and 10 attributes, then rules are all that are possible, i.e. 5<sup>10</sup>.
    - But there are techniques that use clustering to reduce the rules considerably based on structure in data.
    - This will ultimately reduce our computation and thus seems a nice thing to try.

-   Second thing is about generation of membership functions
    - Currently they are generated uniformly, using grid partition.
    - The idea in paper was to generate them randomly, but here is the catch,
        - In ELM, the weights were randomly chosen because they allow the output side to get more variety of abstractions from inputs, from which the appropriate outputs can be learned.
        - But, here things are different, randomly choosing parameters of membership functions can be a bad representation of input data.
        - What we can try is, to use FCM clustering (genfis3) or genfis2. This seem a better choice for starting with.