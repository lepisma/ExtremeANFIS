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

test_1 = [test, [ones(10, 1); zeros(20, 1)]];
test_2 = [test, [zeros(10, 1); ones(10, 1); zeros(10, 1)]];

% Constants
n_mfs = 3;
anfis_iter = 10;
elanfis_iter = 50;

% Training anfis
anfis_1 = anfis(train_1, n_mfs, anfis_iter);
anfis_2 = anfis(train_2, n_mfs, anfis_iter);

% Training elanfis
elanfis_1 = sir.elanfis(train_1(:, 1:end-1), train_1(:, end), n_mfs, elanfis_iter, test_1(:, 1:end-1), test_1(:, end));
elanfis_2 = sir.elanfis(train_2(:, 1:end-1), train_2(:, end), n_mfs, elanfis_iter, test_2(:, 1:end-1), test_2(:, end));
