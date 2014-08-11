%Purpose : To train parameters of Extreme-ANFIS

% The function extremeANFIS requires following arguments:
% 1) trainData => training data in the form of columns % {training data -> [input_1 input_2 ... input_n , output_1, output_2, ... output_k]
%   size=>(Number of sample)X(n+k)
% 2) nMemberFunction => number of membership functions [default=2]
% 3) nOutput => number of outputs in training data [default=1]
% 4) nEpoches => number of epoches [default=50]

%The output of function are:
% 1) finalRMSE => root mean square error value of final designed structure
% 2) Parameters => Structure containing trained parameters

function [finalRMSE, Parameters] = extremeanfis(trainData,nMembershipFn,nOutput,nEpoches)
if nargin <= 3,
    nEpoches=50;
end
if nargin <= 2,
    nOutput=1;
end
if nargin <= 1,
    nMembershipFn=2;
end

finalRMSE=1000;
nInputs=numel(trainData(1,:))-nOutput;
minData=min(trainData);
rangeInput=max(trainData)-minData; %finding ranges of every input
ctr2ctrDist= rangeInput/(nMembershipFn-1);
for Epoches=1:nEpoches,
    % Estimating random mf parameters
    for j=1:1:nInputs,
        for i=1:1:nMembershipFn,
            c(j,i)=((rand-0.5)*ctr2ctrDist(j))+(minData(j)+(i-1)*ctr2ctrDist(j));
            a(j,i)=(rand-0.5)*rangeInput(j)/(2*nMembershipFn-2)*2+rangeInput(j)/(2*nMembershipFn-2);
            b(j,i)=rand*.2+1.9;
        end
    end
    
    %Calculating membership grades
    for m=1:length(trainData),
        for j=1:nInputs,
            for i=1:1:nMembershipFn,
                membershipGrades(j,i)=1/(1+(abs((trainData(m,j)-c(j,i))/a(j,i)))^(2*b(j,i)));
            end
        end
        
        %Calculating firing strength
        for i=1:nInputs,
            t=1;
            for k=1:nMembershipFn^(i-1),
                for j=1:nMembershipFn,
                    for l=1:nMembershipFn^(nInputs-i);
                        B(i,t)=membershipGrades(i,j);
                        t=t+1;
                    end
                end
            end
        end
        
        weights=prod(B);
        
        %Calculating Normalised Firing
        weightNormalize=weights/sum(weights);
        
        %Generating X of f=XZ
        for j=1:1:nInputs,
            X1(j,:)= trainData(m,j)*weightNormalize;
        end
        X1(nInputs+1,:)= weightNormalize;
        X(m,:)=reshape(X1,1,[]);
    end
    
    %Evaluating consequent parameters(Z) for first output by solving linear equation
    for i=1:nOutput,
        Z(i).ZZ= X\trainData(:,nInputs+i);
        Z(i).z=reshape(Z(i).ZZ,nInputs+1,nMembershipFn^nInputs)';
    end
    % Finding error for each training pair
    SumSqrErr(:).SSE=0;
    for m=1:length(trainData),
        for i=1:nOutput,
            op(i).op=X(m,:)*Z(i).ZZ;
            error(i).err(m)=trainData(m,nInputs+i)-op(i).op;
        end
    end
    finalSum=0;
    for i=1:nOutput,
        finalSum=finalSum+sum((error(i).err(:))'.^2);
    end
    
    RMSE=sqrt(finalSum/(length(trainData)*nOutput));
    
    if RMSE<finalRMSE,
        finalRMSE=RMSE;
        Parameters.c=c;
        Parameters.a=a;
        Parameters.b=b;
        for i=1:nOutput,
            Parameters.con(i).consequent=Z(i).z;
        end
    end
end

end

