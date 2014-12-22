% Classification test for banknote data
% 2 classes, training 1 classifier
clear all; close all; clc;
addpath '../../packages/';

data = importdata('data_banknote_authentication.txt');

% Removing extreme values from getting into testing data
[extremes, data] = util.remove_extremes(data);

% Leaving 200 for testing (100 for each class)
test = [data(1:100, :); data(end-99:end, :)];
train = [data(101:end-100, :); extremes];

% Constants
n_mfs = 3;
anfis_iter = 10;
elanfis_iter = 50;

% Training anfis
afis = anfis(train, n_mfs, anfis_iter);

% Training elanfis
elfis = sir.elanfis(train(:, 1:end-1), train(:, end), n_mfs, elanfis_iter, test(:, 1:end-1), test(:, end));

% Training exanfis (diagnostic purpose)
exfis = extreme.exanfis(train, n_mfs, elanfis_iter, test);

% Training zfis
zfis = zfis.zfis(train, n_mfs, elanfis_iter, test);

% Testing

anfis_out = evalfis(test(:, 1:end-1), afis);

elanfis_out = evalfis(test(:, 1:end-1), elfis);

exanfis_out = evalfis(test(:, 1:end-1), exfis);

zfis_out = evalfis(test(:, 1:end-1), zfis);

% Thresholding
% Note : Reusing ova_clear (but it adds extra col, so will remove it later)
e_out = util.ova_clear(elanfis_out);
a_out = util.ova_clear(anfis_out);
ex_out = util.ova_clear(exanfis_out);
z_out = util.ova_clear(zfis_out);

% Printing percentage error in each method
anfis_err = sum(sum(abs(test(:, end) - a_out(:, 1)))) * 50 / size(test, 1)
elanfis_err = sum(sum(abs(test(:, end) - e_out(:, 1)))) * 50 / size(test, 1)
exanfis_err = sum(sum(abs(test(:, end) - ex_out(:, 1)))) * 50 / size(test, 1)
zfis_err = sum(sum(abs(test(:, end) - z_out(:, 1)))) * 50 / size(test, 1)