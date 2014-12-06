% Classification test for iriss
% 3 classes, training 2 classifiers
clear all; close all; clc;
addpath '../../packages/';

% Load data
load fisheriris
train = [meas(1:40, :); meas(51:90, :); meas(101:140, :)];
test = [meas(41:50, :); meas(91:100, :); meas(141:150, :)];

% Preparing data for one-vs-all classifiers
train_1 = [train, [ones(40, 1); zeros(80, 1)]]; % Trains for class 1
train_2 = [train, [zeros(40, 1); ones(40, 1); zeros(40, 1)]]; % Trains for class 2

test_out = [[ones(10, 1); zeros(20, 1)], ...
            [zeros(10, 1); ones(10, 1); zeros(10, 1)], ...
            [zeros(20, 1); ones(10, 1)]];

% Constants
n_mfs = 4;
anfis_iter = 10;
elanfis_iter = 50;

% Training anfis
anfis_1 = anfis(train_1, n_mfs, anfis_iter);
anfis_2 = anfis(train_2, n_mfs, anfis_iter);

% Training elanfis
elanfis_1 = sir.elanfis(train_1(:, 1:end-1), train_1(:, end), n_mfs, elanfis_iter, test, test_out(:, 1));
elanfis_2 = sir.elanfis(train_2(:, 1:end-1), train_2(:, end), n_mfs, elanfis_iter, test, test_out(:, 2));

% Training exanfis (diagnostic purpose)
exanfis_1 = extreme.exanfis(train_1, n_mfs, elanfis_iter, [test, test_out(:, 1)]);
exanfis_2 = extreme.exanfis(train_2, n_mfs, elanfis_iter, [test, test_out(:, 2)]);

% Testing

anfis_out = zeros(size(test, 1), 2);
elanfis_out = zeros(size(test, 1), 2);
exanfis_out = zeros(size(test, 1), 2);

anfis_out(:, 1) = evalfis(test, anfis_1);
anfis_out(:, 2) = evalfis(test, anfis_2);

elanfis_out(:, 1) = evalfis(test, elanfis_1);
elanfis_out(:, 2) = evalfis(test, elanfis_2);

exanfis_out(:, 1) = evalfis(test, exanfis_1);
exanfis_out(:, 2) = evalfis(test, exanfis_2);

% Resolving classes
e_out = util.ova_clear(elanfis_out);
a_out = util.ova_clear(anfis_out);
ex_out = util.ova_clear(exanfis_out);

% Printing percentage error in each method
anfis_err = sum(sum(abs(test_out - a_out))) * 100 / size(test, 1)
elanfis_err = sum(sum(abs(test_out - e_out))) * 100 / size(test, 1)
exanfis_err = sum(sum(abs(test_out - ex_out))) * 100 / size(test, 1)