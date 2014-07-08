% Binary classification task with iris dataset

dat = importdata('../data/iris.csv');
dat = dat.data;

[~, min_indices] = min(dat);
[~, max_indices] = max(dat);

data_without_extremes = dat(setdiff(1:size(dat, 1), [min_indices max_indices]), :);
test_indices = randsample(1:size(data_without_extremes, 1), 80);

data_test = dat(test_indices, :);

data_train = data_without_extremes(setdiff(1:size(data_without_extremes, 1), test_indices), :);

data_train = [data_train; dat([min_indices max_indices], :)];

% Testing

n_mfs = 2;

addpath('../packages/');

fis = extreme.exanfis(data_train, n_mfs, 2, data_test);

err = extreme.error(fis, data_test)