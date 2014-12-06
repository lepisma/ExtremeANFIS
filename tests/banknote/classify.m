% Classification test for banknote data
% 2 classes, training 1 classifier
clear all; close all; clc;
addpath '../../packages/';

data = importdata('data_banknote_authentication.txt');

% Leaving 200 for testing (100 for each class)
train = [data(1:510, :); data(611:1272, :)];
test = [data(511:610, :); data(1273:1372, :)];

% Constants
n_mfs = 4;
anfis_iter = 10;
elanfis_iter = 50;

% Training anfis
afis = anfis(train, n_mfs, anfis_iter);

% Training elanfis
elfis = sir.elanfis(train(:, 1:end-1), train(:, end), n_mfs, elanfis_iter, test(:, 1:end-1), test(:, end));

% Training exanfis (diagnostic purpose)
exfis = extreme.exanfis(train, n_mfs, elanfis_iter, test);

% Testing

anfis_out = evalfis(test(:, 1:end-1), afis);

elanfis_out = evalfis(test(:, 1:end-1), elfis);

exanfis_out = evalfis(test(:, 1:end-1), exfis);

% Thresholding
% Note : Reusing ova_clear (but it adds extra col, so will remove it later)
e_out = util.ova_clear(elanfis_out);
a_out = util.ova_clear(anfis_out);
ex_out = util.ova_clear(exanfis_out);

% Printing percentage error in each method
anfis_err = sum(sum(abs(test(:, end) - a_out(:, 1)))) * 100 / size(test, 1)
elanfis_err = sum(sum(abs(test(:, end) - e_out(:, 1)))) * 100 / size(test, 1)
exanfis_err = sum(sum(abs(test(:, end) - ex_out(:, 1)))) * 100 / size(test, 1)