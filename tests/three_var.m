% A function with three variables

x = 1 : 6;
x = repmat(x, 36, 1);
x = x(:);

y = 1 : 6;
y = repmat(y, 6, 1);
y = y(:);
y = repmat(y, 6, 1);

z = (1 : 6)';
z = repmat(z, 36, 1);

o = (1 + x .^ 0.5 + y .^ (-1) + z .^ (-1.5)).^ 2;
data_train = [x y z o];

%----

x = 1.5 : 5.5;
x = repmat(x, 25, 1);
x = x(:);

y = 1.5 : 5.5;
y = repmat(y, 5, 1);
y = y(:);
y = repmat(y, 5, 1);

z = (1.5 : 5.5)';
z = repmat(z, 25, 1);

o = (1 + x .^ 0.5 + y .^ (-1) + z .^ (-1.5)).^ 2;
data_test = [x y z o];

% Data generation done

% Adding package path
addpath('../packages/');

n_mfs = 8;

% Training ANFIS
a_fis = anfis(data_train, n_mfs, 20);

% Training Extreme ANFIS
e_fis = extreme.exanfis(data_train, n_mfs, 60, data_test);

% Training Genetic Extreme ANFIS
% ga_fis = genetic.ganfis(data_train, n_mfs, data_test);

% Training Bagged Extreme ANFIS
%ba_fisses = ensemble.bagging.bagfis(data_train, n_mfs, 60, 300);

% Training Additive Extreme ANFIS
% add_fisses = ensemble.additive.addfis(data_train, n_mfs, data_test);

% Errors ANFIS
a_fis_train = extreme.rmse(a_fis, data_train)
a_fis_test = extreme.rmse(a_fis, data_test)

% Errors Extreme ANFIS
e_fis_train = extreme.rmse(e_fis, data_train)
e_fis_test = extreme.rmse(e_fis, data_test)

% Errors Genetic Extreme ANFIS
% ga_fis_train = genetic.rmse(ga_fis, data_train)
% ga_fis_test = genetic.rmse(ga_fis, data_test)

% Errors Bagged Extreme ANFIS
%ba_fisses_train = ensemble.bagging.rmse(ba_fisses, data_train)
%ba_fisses_test = ensemble.bagging.rmse(ba_fisses, data_test)

%weights = ensemble.bagging.post_bag_elm(ba_fisses, data_train);

%ba_fisses_train2 = ensemble.bagging.rmse2(ba_fisses, weights, data_train)
%ba_fisses_test2 = ensemble.bagging.rmse2(ba_fisses, weights, data_test)

% Errors Additive Extreme ANFIS
% add_fisses_train = ensemble.additive.rmse(add_fisses, data_train)
% add_fisses_test = ensemble.additive.rmse(add_fisses, data_test)