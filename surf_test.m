% This is a bit more complicated function

n_pts = 51;
x = 1:6;
y = x;
z = x;
w = x;

o = (1 + x .^ 0.5 + y .^ (-1) + z .^ (-1.5) + w .^ (-2.5)).^ 5;
data_train = [x' y' z' w' o'];

n_mfs = 5;

a_fis = anfis(data_train, n_mfs);

e_fis = exanfis(data_train(:, 1:4), o, n_mfs);