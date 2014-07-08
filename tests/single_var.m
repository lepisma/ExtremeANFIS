% Function with single variable

n_pts = 501;
x = linspace(-1, 1, n_pts)';
y = 0.6 * sin(pi * x) + 0.3 * sin(3 * pi * x) + 0.1 * sin(5 * pi * x);
data = [x y];
data_train = data(1 : 2 : n_pts, :);
data_test = data(2 : 2 : n_pts, :);

n_mfs = 10;

addpath('../packages/');
% Training ANFIS
a_fis = anfis(data_train, n_mfs, 20);

% Training Extreme ANFIS
e_fis = extreme.exanfis(data_train, n_mfs, 20, data_test);

% Training Genetic Extreme ANFIS
ga_fis = genetic.ganfis(data_train, n_mfs, data_test);

% Training Bagged Extreme ANFIS
ba_fisses = ensemble.bagging.bagfis(data_train, n_mfs, 20, 200);

% Training Additive Extreme ANFIS
add_fisses = ensemble.additive.addfis(data_train, n_mfs, data_test);

% Errors ANFIS
a_fis_train = extreme.rmse(a_fis, data_train)
a_fis_test = extreme.rmse(a_fis, data_test)

% Errors Extreme ANFIS
e_fis_train = extreme.rmse(e_fis, data_train)
e_fis_test = extreme.rmse(e_fis, data_test)

% Errors Genetic Extreme ANFIS
ga_fis_train = genetic.rmse(ga_fis, data_train)
ga_fis_test = genetic.rmse(ga_fis, data_test)

% Errors Bagged Extreme ANFIS
ba_fisses_train = ensemble.bagging.rmse(ba_fisses, data_train)
ba_fisses_test = ensemble.bagging.rmse(ba_fisses, data_test)

% Errors Additive Extreme ANFIS
add_fisses_train = ensemble.additive.rmse(add_fisses, data_train)
add_fisses_test = ensemble.additive.rmse(add_fisses, data_test)