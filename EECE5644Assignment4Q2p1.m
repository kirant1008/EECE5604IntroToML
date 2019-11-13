clc;
clear all;
close all;

%Generation of DataSet
no_Samples=1000;
p_q1=0.35;p_q2=0.65;
%Mean and Covariance for class 1
mean=[0;0];cov=eye(2);

%Generation of Dataset
class_Priors=[p_q1,p_q2];
prior_threshold=[0,cumsum(class_Priors)];%inorder to generate datasets
prob_uni=rand(1,no_Samples);%generating 10000 samples of probablities
og_Labels=zeros(1,no_Samples);

for i=1:2
    pntr=find(prob_uni>=prior_threshold(i) &  prob_uni<=prior_threshold(i+1));
    og_Labels(1,pntr)=i*ones(1,length(pntr));
    count_samples(1,i)=length(pntr);
    if i == 1
    data(:,pntr)=mvnrnd(mean,cov,length(pntr))';
    else
    %Radius and Theta for class 2
        r1=2;r2=3;
        r = r1 + (r2-r1).*rand(length(pntr),1);
        tht1=-pi;tht2=pi;
        theta = tht1+(tht2-tht1).*rand(length(pntr),1);
        x=r.*cos(theta);
        y=r.*sin(theta);
        con = cat(2,x,y);
        data(:,pntr)=con';
    end
    figure(1)
    plot(data(1,pntr),data(2,pntr),'.'); axis equal, hold on,        
end
hold off
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+'},'Location','northeast')

dataSet = cat(2,data',og_Labels');

count_error=0;

model2=fitcsvm(dataSet(:,1:2),dataSet(:,3),'KernelFunction','linear','OptimizeHyperparameters','BoxConstraint');
mdl1=crossval(model2,'KFold',10);
min_loss1=kfoldLoss(mdl1);

model2=fitcsvm(dataSet(:,1:2),dataSet(:,3),'KernelFunction','gaussian','OptimizeHyperparameters','auto');
mdl2=crossval(model2,'KFold',10);
min_loss2=kfoldLoss(mdl2);
