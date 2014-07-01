% Binary classification task

A = importdata('data/A.csv');
B = importdata('data/A.csv');

A = [A ones(size(A, 1), 1)];
B = [B zeros(size(B, 1), 1)];

[~, min_indices] = min(A);
[~, max_indices] = max(A);
A_without_extremes = A(setdiff(1:size(A, 1), [min_indices max_indices]), :);

test_indices = randsample(1:size(A_without_extremes, 1), 200);
A_test = A_without_extremes(test_indices, :);
A_train = [A_without_extremes(setdiff(1:size(A_without_extremes, 1), test_indices), :); A([min_indices max_indices], :)];

[~, min_indices] = min(B);
[~, max_indices] = max(B);
B_without_extremes = B(setdiff(1:size(B, 1), [min_indices max_indices]), :);

test_indices = randsample(1:size(B_without_extremes, 1), 200);
B_test = B_without_extremes(test_indices, :);
B_train = [B_without_extremes(setdiff(1:size(B_without_extremes, 1), test_indices), :); B([min_indices max_indices], :)];

data_train = [A_train; B_train];
data_test = [A_test; B_test];

% Testing

n_mfs = 3;

[e_fis, e_errs] = exanfis(data_train(:, 10:17), n_mfs, 2, data_test(:, 10:17));
%a_fis = anfis(data_train(:, 13:17), n_mfs, 10);

% Train errors
e_err_train = rmse(e_fis, data_train(:, 10:17))
%a_err_train = rmse(a_fis, data_train(:, 13:17))

% Test errors
e_err_test = rmse(e_fis, data_test(:, 10:17))
%a_err_test = rmse(a_fis, data_test(:, 13:17))