% Classification test for balance scale dataset
% 3 classes, training 2 classifiers
% dataset has 288 'L', 288 'R', 49 'B' classes 

clear all; close all; clc;
addpath '../../packages/';

raw_data = importdata('balance-scale.data.txt');

balanced_idx = strcmp(raw_data.textdata, 'B');
left_idx = strcmp(raw_data.textdata, 'L');
right_idx = strcmp(raw_data.textdata, 'R');

% sorting data in order 'L', 'R', 'B'
data = [raw_data.data(left_idx, :); raw_data.data(right_idx, :); raw_data.data(balanced_idx, :)];

% test train split (60, 60, 10)
test = [data(1:60, :); data(289:348, :); data(end-9:end, :)];
train = [data(61:288, :); data(349:end-49, :); data(end-48:end-10, :)];

% data for classifier 1 (left)
train_1 = [train, [ones(228, 1); zeros(267, 1)]];
% data for classifier 2 (right)
train_2 = [train, [zeros(228, 1); ones(228, 1); zeros(39, 1)]];

test_out = [[ones(60, 1); zeros(70, 1)], ...
            [zeros(60, 1); ones(60, 1); zeros(10, 1)], ...
            [zeros(120, 1); ones(10, 1)]];
        
% Constants
n_mfs = 3;
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