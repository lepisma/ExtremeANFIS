% Classification test for iris
% 3 classes, training 3 classifiers
clear all; close all; clc;
addpath '../../packages/';

% Load data
load fisheriris
meas = util.normalize(meas);

data = [meas [ones(50, 1); 2 * ones(50, 1); 3 * ones(50, 1)]];
[test, train, n_test, ~] = util.test_train_split(data, 0.25);

% Preparing data for one-vs-all classifiers
train_1 = [train(:, 1:end-1), train(:, end) == 1];
train_2 = [train(:, 1:end-1), train(:, end) == 2];
train_3 = [train(:, 1:end-1), train(:, end) == 3];

% Removing output from test
test = test(:, 1:end-1);

test_out = [[ones(n_test(1), 1); zeros(n_test(2) + n_test(3), 1)], ...
            [zeros(n_test(1), 1); ones(n_test(2), 1); zeros(n_test(3), 1)], ...
            [zeros(n_test(1) + n_test(2), 1); ones(n_test(3), 1)]];

% Constants
n_mfs = 4;
anfis_iter = 10;
elanfis_iter = 50;

% Training anfis
anfis_1 = anfis(train_1, n_mfs, anfis_iter);
anfis_2 = anfis(train_2, n_mfs, anfis_iter);
anfis_3 = anfis(train_3, n_mfs, anfis_iter);

% Training elanfis
elanfis_1 = el.elanfis(train_1(:, 1:end-1), train_1(:, end), n_mfs, elanfis_iter, test, test_out(:, 1));
elanfis_2 = el.elanfis(train_2(:, 1:end-1), train_2(:, end), n_mfs, elanfis_iter, test, test_out(:, 2));
elanfis_3 = el.elanfis(train_3(:, 1:end-1), train_3(:, end), n_mfs, elanfis_iter, test, test_out(:, 3));

% Training exanfis (diagnostic purpose)
exanfis_1 = extreme.exanfis(train_1, n_mfs, elanfis_iter, [test, test_out(:, 1)]);
exanfis_2 = extreme.exanfis(train_2, n_mfs, elanfis_iter, [test, test_out(:, 2)]);
exanfis_3 = extreme.exanfis(train_3, n_mfs, elanfis_iter, [test, test_out(:, 3)]);

% Training zanfis
zfis_1 = zfis.zfis(train_1, n_mfs, elanfis_iter, [test, test_out(:, 1)]);
zfis_2 = zfis.zfis(train_2, n_mfs, elanfis_iter, [test, test_out(:, 2)]);
zfis_3 = zfis.zfis(train_3, n_mfs, elanfis_iter, [test, test_out(:, 3)]);

% Testing

anfis_out = zeros(size(test, 1), 3);
elanfis_out = zeros(size(test, 1), 3);
exanfis_out = zeros(size(test, 1), 3);
zfis_out = zeros(size(test, 1), 3);

anfis_out(:, 1) = evalfis(test, anfis_1);
anfis_out(:, 2) = evalfis(test, anfis_2);
anfis_out(:, 3) = evalfis(test, anfis_3);

elanfis_out(:, 1) = evalfis(test, elanfis_1);
elanfis_out(:, 2) = evalfis(test, elanfis_2);
elanfis_out(:, 3) = evalfis(test, elanfis_3);

exanfis_out(:, 1) = evalfis(test, exanfis_1);
exanfis_out(:, 2) = evalfis(test, exanfis_2);
exanfis_out(:, 3) = evalfis(test, exanfis_3);

zfis_out(:, 1) = evalfis(test, zfis_1);
zfis_out(:, 2) = evalfis(test, zfis_2);
zfis_out(:, 3) = evalfis(test, zfis_3);


% Resolving classes
e_out = util.ova_clear(elanfis_out);
a_out = util.ova_clear(anfis_out);
ex_out = util.ova_clear(exanfis_out);
z_out = util.ova_clear(zfis_out);

% Printing percentage error in each method
anfis_err = sum(sum(abs(test_out - a_out))) * 50 / size(test, 1)
elanfis_err = sum(sum(abs(test_out - e_out))) * 50 / size(test, 1)
exanfis_err = sum(sum(abs(test_out - ex_out))) * 50 / size(test, 1)
zfis_err = sum(sum(abs(test_out - z_out))) * 50 / size(test, 1)

% RMSE
err = abs(anfis_out - test_out);
rms(err(:))
err = abs(elanfis_out - test_out);
rms(err(:))
err = abs(exanfis_out - test_out);
rms(err(:))
err = abs(zfis_out - test_out);
rms(err(:))
