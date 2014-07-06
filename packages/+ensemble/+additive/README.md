## Additive Ensemble of Extreme ANFIS

The functions here make an ensemble of two **Extreme ANFIS**.
- The first one is a simple **Extreme ANFIS** working to predict the output on training data.
- The second one works on testing data and trains another **Extreme ANFIS** that outputs the error in prediction of first model.
- When combined, the results are expected to reduce the error and increase the speed as compared to **Extreme ANFIS**, since only two trainings are needed.


#### To do
- Classificaton error file.
- Add bootstrapping to add more than two models. (If the performance is poor)
