% Classification test

% -------------------- WINE
%data = importdata('winedata.txt');

%data = [data(:, 2:14), data(:, 1)];

%train = [data(1:49, :); data(60:120, :); data(131:168, :)];
%test = [data(50:59, :); data(121:130, :); data(169:178, :)];

% Class training
%train_1 = [train(:, 1:13), [ones(49, 1); zeros(99, 1)]];
%train_2 = [train(:, 1:13), [zeros(49, 1); ones(61, 1); zeros(38, 1)]];
%train_3 = [train(:, 1:13), [zeros(110, 1); ones(38, 1)]];

% -------------------- IRIS
load fisheriris

train = [meas(1:40, :); meas(51:90, :); meas(101:140, :)];
test = [meas(41:50, 1:4); meas(91:100, 1:4); meas(141:150, 1:4)];

% Class training
train_1 = [train(:, 1:4), [ones(40, 1); zeros(80, 1)]];
train_2 = [train(:, 1:4), [zeros(40, 1); ones(40, 1); zeros(40, 1)]];
train_3 = [train(:, 1:4), [zeros(80, 1); ones(40, 1)]];

out_1 = [ones(10, 1); zeros(20, 1)];
out_2 = [zeros(10, 1); ones(10, 1); zeros(10, 1)];
out_3 = [zeros(20, 1); ones(10, 1)];

n_mfs = 4;

infis_1 = genfis1(train_1, n_mfs, 'gbellmf');
infis_2 = genfis1(train_2, n_mfs, 'gbellmf');
infis_3 = genfis1(train_3, n_mfs, 'gbellmf');

afis_1 = anfis(train_1, infis_1, 10);
afis_2 = anfis(train_2, infis_2, 10);
afis_3 = anfis(train_3, infis_3, 10);

efis_1 = sir.elanfis(train_1(:, 1:4), train_1(:, 5), n_mfs, 80);
efis_2 = sir.elanfis(train_2(:, 1:4), train_2(:, 5), n_mfs, 80);
efis_3 = sir.elanfis(train_3(:, 1:4), train_3(:, 5), n_mfs, 80);

exfis_1 = extreme.exanfis(train_1, n_mfs, 80, [test, out_1]);
exfis_2 = extreme.exanfis(train_2, n_mfs, 80, [test, out_2]);
exfis_3 = extreme.exanfis(train_3, n_mfs, 80, [test, out_3]);

a_1_out = evalfis(test(:, 1:4), afis_1);
a_2_out = evalfis(test(:, 1:4), afis_2);
a_3_out = evalfis(test(:, 1:4), afis_3);
e_1_out = evalfis(test(:, 1:4), efis_1);
e_2_out = evalfis(test(:, 1:4), efis_2);
e_3_out = evalfis(test(:, 1:4), efis_3);
ex_1_out = evalfis(test(:, 1:4), exfis_1);
ex_2_out = evalfis(test(:, 1:4), exfis_2);
ex_3_out = evalfis(test(:, 1:4), exfis_3);

a_rmse = sqrt(sum((a_1_out - out_1) .^ 2)) + sqrt(sum((a_2_out - out_2) .^ 2)) + sqrt(sum((a_3_out - out_3) .^ 2))
e_rmse = sqrt(sum((e_1_out - out_1) .^ 2)) + sqrt(sum((e_2_out - out_2) .^ 2)) + sqrt(sum((e_3_out - out_3) .^ 2))
ex_rmse = sqrt(sum((ex_1_out - out_1) .^ 2)) + sqrt(sum((ex_2_out - out_2) .^ 2)) + sqrt(sum((ex_3_out - out_3) .^ 2))

for i = 1:30
    if a_1_out(i) < 0.5
        a_1_out(i) = 0;
    else
        a_1_out(i) = 1;
    end
    
    if a_2_out(i) < 0.5
        a_2_out(i) = 0;
    else
        a_2_out(i) = 1;
    end
    
    if a_3_out(i) < 0.5
        a_3_out(i) = 0;
    else
        a_3_out(i) = 1;
    end
    
    if e_1_out(i) < 0.5
        e_1_out(i) = 0;
    else
        e_1_out(i) = 1;
    end
    
    if e_2_out(i) < 0.5
        e_2_out(i) = 0;
    else
        e_2_out(i) = 1;
    end
    
    if e_3_out(i) < 0.5
        e_3_out(i) = 0;
    else
        e_3_out(i) = 1;
    end

    if ex_1_out(i) < 0.5
        ex_1_out(i) = 0;
    else
        ex_1_out(i) = 1;
    end 
    
    if ex_2_out(i) < 0.5
        ex_2_out(i) = 0;
    else
        ex_2_out(i) = 1;
    end
    
    if ex_3_out(i) < 0.5
        ex_3_out(i) = 0;
    else
        ex_3_out(i) = 1;
    end

end

a_error_count = sum(a_1_out ~= out_1) + sum(a_2_out ~= out_2) + sum(a_3_out ~= out_3);
e_error_count = sum(e_1_out ~= out_1) + sum(e_2_out ~= out_2) + sum(e_3_out ~= out_3);
ex_error_count = sum(ex_1_out ~= out_1) + sum(ex_2_out ~= out_2) + sum(ex_3_out ~= out_3);

% a_error_count = sum(a_2_out ~= out_2);
% e_error_count = sum(e_2_out ~= out_2);
% ex_error_count = sum(ex_2_out ~= out_2);

a_error = 100 * (a_error_count / 30)
e_error = 100 * (e_error_count / 30)
ex_error = 100 * (ex_error_count / 30)
