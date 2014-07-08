## Bagging of Extreme ANFIS

The functions here make an ensemble of **Extreme ANFIS** using bagging.

- For each weak learner, the training data is created from the global training data.
- Each learner learns and the output is calculated by averaging the output from each learner.

### To do

**Finding Output**

- Averaging (current).
- Combining according to error.
- Combining using Matrix inverse (another ELM).
