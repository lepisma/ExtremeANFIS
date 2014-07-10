% Binary classification task

A = importdata('../data/A.csv');
B = importdata('../data/B.csv');

A = [A ones(size(A, 1), 1)];
B = [B zeros(size(B, 1), 1)];

[~, min_indices] = min(A);
[~, max_indices] = max(A);
A_without_extremes = A(setdiff(1:size(A, 1), [min_indices max_indices]), :);

test_indices = randsample(1:size(A_without_extremes, 1), 200);
A_test = A_without_extremes(test_indices, :);
%A_train = [A_without_extremes(setdiff(1:size(A_without_extremes, 1), test_indices), :); A([min_indices max_indices], :)];
A_train = A_without_extremes(setdiff(1:size(A_without_extremes, 1), test_indices), :);
A_extreme = A([min_indices max_indices], :);

[~, min_indices] = min(B);
[~, max_indices] = max(B);
B_without_extremes = B(setdiff(1:size(B, 1), [min_indices max_indices]), :);

test_indices = randsample(1:size(B_without_extremes, 1), 200);
B_test = B_without_extremes(test_indices, :);
%B_train = [B_without_extremes(setdiff(1:size(B_without_extremes, 1), test_indices), :); B([min_indices max_indices], :)];
B_train = B_without_extremes(setdiff(1:size(B_without_extremes, 1), test_indices), :);
B_extreme = B([min_indices max_indices], :);

data_train = [A_train; B_train];
data_test = [A_test; B_test];
data_extreme = [A_extreme; B_extreme];
data_train = [data_train; data_extreme];

% Testing

n_mfs = 3;

addpath('../packages/');

fis = extreme.exanfis(data_train(:, 13 : 17), n_mfs, 20, data_test(:, 13 : 17));

err = extreme.error(fis, data_test(:, 13 : 17))
