% Classification test for vertebral column dataset
% 3 classes, training 2 classifiers 
%
% Dataset has been modified to add column names (A, B, C . . . G)
% 6 features, 3 classes and 310 instances
% Classes are 'DH', 'SL' and 'NO' with frequency of 60, 150 and 100

clear all; close all; clc;
addpath '../../packages/';

data = readtable('vertebral.dat', 'Delimiter' ,' ');

% Classes sorted in order
DH_idx = strcmp(data.G, 'DH');
SL_idx = strcmp(data.G, 'SL');
NO_idx = strcmp(data.G, 'NO');

% Filtering numbers
data = table2array(data(:, 1:6));

% Test train split (12, 30, 20)
test = [data(1:12, :); data(61:90, :); data(211:230, :)];
train = [data(13:60, :); data(91:210, :); data(231:310, :)];

% data for classifier 1 ('DH')
train_1 = [train, [ones(48, 1); zeros(200, 1)]];
% data for classifier 2 ('SL')
train_2 = [train, [zeros(48, 1); ones(120, 1); zeros(80, 1)]];

test_out = [[ones(12, 1); zeros(50, 1)], ...
            [zeros(12, 1); ones(30, 1); zeros(20, 1)], ...
            [zeros(42, 1); ones(20, 1)]];
        
% Constants
n_mfs = 2;
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