
clc;
clear;
close all;

%Values of Dataset
no_Samples=999;
mean(:,1)=[0;0]; cov(:,:,1)=[2 ,0.5 ; 0.5 ,1]; c1_pw=0.3;%for class 1 
mean(:,2)=[2;2] ; cov(:,:,2)=[2 ,-1.9 ; -1.9 ,5];c2_pw=0.7 ;%for class 2 


class_Priors=[c1_pw,c2_pw];
prior_threshold=[0,cumsum(class_Priors)];%inorder to generate datasets
prob_uni=rand(1,no_Samples);%generating 999 samples of probablities
og_Labels=zeros(1,no_Samples);


figure(1)
%generation of dataset
for i=1:2
    pntr=find(prob_uni>=prior_threshold(i) &  prob_uni<=prior_threshold(i+1));
    og_Labels(1,pntr)=(i-1)*ones(1,length(pntr));
    count_samples(1,i)=length(pntr);
    data(:,pntr)=mvnrnd(mean(:,i),cov(:,:,i),length(pntr))';
    figure(1)
    subplot(2,1,1)
    plot(data(1,pntr),data(2,pntr),'.'); axis equal, hold on,
end
hold off
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+'},'Location','northeast')

%Re-Arranginf Data-Set
ent_data=cat(2,data',og_Labels');

%Dividing in Training and Test Data set
train_x=ent_data(1:999,1:2);
train_y=ent_data(1:999,3);

[m, n] = size(train_x);


% Add intercept term to x and X_test
train_x= [ones(m, 1) train_x];

% Initialize fitting parameters
init_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc is used to obtain optimal theta 
[theta, cost] = fminunc(@(grad)(costFunction(grad,train_x,train_y)), init_theta, options);

% Print theta to screen
fprintf('Cost at optimumm theta : %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

est_y=predict(theta,train_x);
subplot(2,1,2)
plot_Data(train_x(:,2:3),est_y(:,1));
title('Plot of Training Samples Classified as Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend('q_+', 'q_-')

%predicting outputs with training data
op = predict(theta,train_x);

%counting Number of errors
error = train_y==op;
countEr=0;
for j=1:999
    if(error(j)==0)
        countEr=countEr+1;
    end
end
p_error=(countEr/999)*100;%probablity of error

%To compute sigmoid of z,logistic regression function
function y = sigmoid(z)
y = 1./(1 + exp(-1*z));
end

%TO compute cost function 
function [cost_J, grad] = costFunction(theta, X, y)

m = length(y);%no of training examples
cost_J=0;

grad = zeros(size(theta));
%Cost function for logistic regression
cost_J = (-1 / m) * sum(y.*log(sigmoid(X * theta)) + (1 - y).*log(1 - sigmoid(X * theta)));
%partial diffrentiation of cost function with respect to Theta
temp = sigmoid (X * theta);
error = temp - y;
grad = (1 / m) * (X' * error); %computing first iter gradient
end

function plot_Data(X, y)

% Find Indices of positive and Negative Examples
ones = find(y==0); 
twos = find(y == 1);
% Plot Examples
plot(X(ones, 1), X(ones, 2),'.');hold on;axis equal;
plot(X(twos, 1), X(twos, 2),'.');hold on;axis equal;

hold off;

end

function p = predict(theta, X)
m = size(X, 1); % Number of training Samples
p = round(sigmoid(X * theta));

end
