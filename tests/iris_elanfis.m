%loading fisheriris data
load fisheriris
addpath('../packages/');


% Generating training data (first 45 samples of each type)
train_y =[[2 * ones(45,1);zeros(45,1);zeros(45,1)] [zeros(45,1);2 * ones(45,1);zeros(45,1)] [zeros(45,1);zeros(45,1);2 * ones(45,1)]];
train_x = [meas(1:45,:); meas(51:95,:); meas(101:145,:)];

fis = sir.elanfis(train_x, train_y, 5, 50);

fis
