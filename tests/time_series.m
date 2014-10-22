% Script for testing performance on chaotic time series

addpath('../packages');

load mgdata.dat

a = mgdata;

time = a(:, 1);
x_t = a(:, 2);

trn_data = zeros(500, 5);
chk_data = zeros(500, 5);

trn_data(:, 1) = x_t(101 : 600);
trn_data(:, 2) = x_t(107 : 606);
trn_data(:, 3) = x_t(113 : 612);
trn_data(:, 4) = x_t(119 : 618);
trn_data(:, 5) = x_t(125 : 624);

chk_data(:, 1) = x_t(601 : 1100);
chk_data(:, 2) = x_t(607 : 1106);
chk_data(:, 3) = x_t(613 : 1112);
chk_data(:, 4) = x_t(619 : 1118);
chk_data(:, 5) = x_t(625 : 1124);

in_fis = genfis1(trn_data, 2);

anfis_out = anfis(trn_data, in_fis, 30);

exanfis_out = extreme.exanfis(trn_data, 2, 30, chk_data);

%bag_out = ensemble.bagging.bagfis(trn_data, 2, 30, 1000);

%add_out = ensemble.additive.addfis(trn_data, 2, chk_data);

extreme.rmse(anfis_out, chk_data)
extreme.rmse(exanfis_out, chk_data)
%ensemble.bagging.rmse(bag_out, chk_data)
%ensemble.additive.rmse(add_out, chk_data)

%weights = ensemble.bagging.post_bag_elm(bag_out, trn_data);
%ensemble.bagging.rmse2(bag_out, weights, chk_data)

input = [trn_data(:, 1:4); chk_data(:, 1:4)];
anfis_output = evalfis(input, anfis_out);
exanfis_output = evalfis(input, exanfis_out);

index = 119:1118;
diffan = x_t(index)-anfis_output;
diffen = x_t(index)-exanfis_output;
plot(time(index), diffan, 'color', 'b'); hold on;
plot(time(index), diffen, 'color', 'r');
xlabel('Time (sec)','fontsize',10);
title('Prediction Errors','fontsize',10);
