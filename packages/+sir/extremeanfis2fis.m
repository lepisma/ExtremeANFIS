%Purpose : To creat Takagi Sugeno FIS structure using Extreme-ANFIS trained
%           Parameters

%The function extremeanfis2fis requires following arguments:
% 1) trainData => as used in extremeanfis command
% 2) Parameters => structure provided by extremeanfis command

%The output of function are:
% 1) OutputFismat => Extreme-ANFIS trained MIMO/MISO/SISO Takagi Sugeno FIS structure for
%                    Simulink use

function [OutputFismat] = extremeanfis2fis(trainData, Parameters)

nInputs=numel(Parameters.a(:,1));
nMembershipFn=numel(Parameters.a(1,:));
OutputFismat=genfis1(trainData(:,1:end-length(Parameters.con)+1),nMembershipFn);
OutputFismat.name='ExtremeANFIS'
for j=1:1:nInputs,
    for i=1:1:nMembershipFn,
        OutputFismat.input(1,j).mf(1,i).params=[Parameters.a(j,i) Parameters.b(j,i) Parameters.c(j,i)];
    end
end
for j=1:length(Parameters.con)
    n=num2str(j);
    OutputFismat.output(1, j).name=['output' n];
    OutputFismat.output(1, j).range=[min(trainData(:,nInputs+j)) max(trainData(:,nInputs+j))];
    for i=1:nMembershipFn^nInputs
        m=num2str(i);
        OutputFismat.output(1, j).mf(1, 1).name=['mf' m];
        OutputFismat.output(1, j).mf(1, i).type='linear';
    end
end
for i=1:nMembershipFn^nInputs
    for j=1:length(Parameters.con)
        OutputFismat.output(1,j).mf(1, i).params=Parameters.con(j).consequent(i,:);
    end
    OutputFismat.rule(1, i).consequent=repmat(OutputFismat.rule(1, i).consequent,1,length(Parameters.con))
end
end

