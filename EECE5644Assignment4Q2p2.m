clc;
clear all;
close all;

%Generation of DataSet
no_Samples=1000;
p_q1=0.35;p_q2=0.65;
%Mean and Covariance for class 1
mean=[0;0];cov=eye(2);

[dt1,og_Labels]=generateData(mean,cov,p_q1,p_q2,no_Samples);
[dt2,og_Labels2]=generateData(mean,cov,p_q1,p_q2,no_Samples);

dataSet = cat(2,dt1',og_Labels');
dataSet2 = cat(2,dt2',og_Labels2');

%for training data
mdl_linear = fitcsvm(dataSet(:,1:2),dataSet(:,3),'KernelFunction','linear','BoxConstraint',0.30075);
mdl_gaussian = fitcsvm(dataSet(:,1:2),dataSet(:,3),'KernelFunction','gaussian','BoxConstraint',21.743,'KernelScale',1.5316);
cls_label_linear=predict(mdl_linear,dataSet(:,1:2));
cls_label_gaussian = predict(mdl_gaussian,dataSet(:,1:2));
%for new training data
cls_label_linear2=predict(mdl_linear,dataSet2(:,1:2));
cls_label_gaussian2 = predict(mdl_gaussian,dataSet2(:,1:2));
figure(2)
p_error1= plot_error(dt1,cls_label_linear,og_Labels)
figure(3)
p_error2= plot_error(dt1,cls_label_gaussian,og_Labels)
figure(4)
p_error3 = plot_error(dt2,cls_label_linear2,og_Labels2)
figure(5)
p_errror4=plot_error(dt2,cls_label_gaussian2,og_Labels2)

% error_linear = plot_error

function p_error = plot_error(dt1,cls_label,og_Labels)
error = cls_label == og_Labels';

count_error=0;
for i=1:1000
    if cls_label(i)==1
        plot(dt1(1,i),dt1(2,i),'.','Color','red'); axis equal;hold on;
    elseif cls_label(i)==2
        plot(dt1(1,i),dt1(2,i),'x','Color','cyan'); axis equal;hold on;
    end
    
    if error(i) == 0
        count_error=count_error+1;
    end
end
hold on
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+'},'Location','northeast')
hold off

p_error=count_error/1000;
end

function [data,og_Labels] = generateData(mean,cov,p_q1,p_q2,no_Samples)
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
end