%Purpose : To simulate final Extreme-ANFIS

%The function extremeanfis2fis requires following arguments:
% 1) data => as used in extremeanfis (Only input columns)
% 2) Parameters => structure provided by extremeanfis command

%The output of function are:
% 1)output => gives final simulated output

function output = simextremeanfis(data, Parameters)
nInputs=numel(Parameters.a(:,1));
nMembershipFn=numel(Parameters.a(1,:));
for m=1:numel(data(:,1)),
    for j=1:1:nInputs,
        for i=1:1:nMembershipFn,
            membershipGrades(j,i)=1/(1+(abs((data(m,j)-Parameters.c(j,i))/Parameters.a(j,i)))^(2*Parameters.b(j,i)));
        end
    end
    %Calculating firing strength
    for i=1:nInputs,
        t=1;
        for k=1:nMembershipFn^(i-1),
            for j=1:nMembershipFn,
                for l=1:nMembershipFn^(nInputs-i),
                    B(i,t)=membershipGrades(i,j);
                    t=t+1;
                end
            end
        end
    end
    
    weights=prod(B);
    
    %Calculating Normalised Firing
    weightNormalize =weights/sum(weights);
    for i=1:length(Parameters.con)
        output(m,i)=sum(weightNormalize.*([data(m,:) 1]*Parameters.con(i).consequent'));
    end
end
end

