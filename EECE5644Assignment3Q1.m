clc;
clear all;
close all;
%Code consists of data generation, evaluating a gaussian(finding the pdf)
%evalGMM(find the log likilihood of each component)
data = generate_random_data(1000)';%Generation of Data set to 1000 
X= split(data);
folds=10;%Number of Folds
%By running the following loop we get probability of each component from
%1 to 6 for particular number of data samples
for m=1:6
for i=1:folds
    [train, test]= kfld(X,i);%gives training and test data for a fold
    model_gmm = fitgmdist(train,m);%fitting a gmm model on train data for m components
    prior=model_gmm.ComponentProportion;%priors
    mean1=model_gmm.mu;%mean
    mean=mean1';
    cov=model_gmm.Sigma;%covariance
    logLikelihood(i) = sum(log(evalGMM(test,prior,mean,cov)));
end
final_mean(m)=sum(logLikelihood)/10;%Final probability for all 6 Components
end

%Dividing data in K folds i.e 9 Trianing and 1 Test dataset
function [ train , test] = kfld(X,n)
train=[];
for f=1:10
    if f~=n
    train=[train;X(:,:,f)];
    end
end
train(n,:)=[];
test=X(:,:,n);
end

%Splitting data into 10 equal parts
function splt = split(data)
 [m n]= size(data);
 ct_inx=1;
for i=1:10
     nxt_ind=(ct_inx+m/10)-1;
     splt(:,:,i)=data(ct_inx:nxt_ind,:);
     ct_inx=nxt_ind;
end
end

function [data ,og_Labels]=  generate_random_data(no_Samples)

mean(:,1)=[-1;0]; cov(:,:,1)=0.1*[10 -4;-4,5]; c1_pw=0.25;%for class 1 
mean(:,2)=[1;0] ; cov(:,:,2)=0.1*[5 0;0,2];c2_pw=0.25 ;%for class 2 
mean(:,3)=[0;1] ; cov(:,:,3)= 0.1*eye(2); c3_pw=0.25 ;%for class 3
mean(:,4)=[0,-1]; cov(:,:,4)= 0.1*[2,0;0,1]; c4_pw=0.25 ;%for class 4

class_Priors=[c1_pw,c2_pw,c3_pw,c4_pw];
prior_threshold=[0,cumsum(class_Priors)];%inorder to generate datasets
prob_uni=rand(1,no_Samples);%generating 10000 samples of probablities
og_Labels=zeros(1,no_Samples);


figure(1)
%generation of dataset
for i=1:4
    pntr=find(prob_uni>=prior_threshold(i) &  prob_uni<=prior_threshold(i+1));
    og_Labels(1,pntr)=i*ones(1,length(pntr));
    count_samples(1,i)=length(pntr);
    data(:,pntr)=mvnrnd(mean(:,i),cov(:,:,i),length(pntr))';
    figure(1)
    plot(data(1,pntr),data(2,pntr),'x'); axis equal, hold on,
end
hold off
title('Plot of IID Samples')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2','W_3','W_4'},'Location','northeast')
end
%In order to find likelihood of each fold
function gmm = evalGMM(x,alpha,mu,Sigma)
x=x';
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%pdf generation for each value of x
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end