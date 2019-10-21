clc;
clear;
close all;
%Generation of iid x using rand 
x = -1 + 2.*rand(10,1);
%Identity matrix 
I=eye(4);

%gamma varying from 10^-1 to 10^1
gamma=10^(-1):1:10^(1);

%Keeping standard deviation fixed at 1
sd=1;

%For Generation of Dataset
for n=1:10
for j=1:10
    v=normrnd(0,sd);%Noise generation for each value of x
    for i=1:10
     
     w(i,:,j,n)=mvnrnd([0 0 0 0],(gamma(i)^2)*I);%Geneartion of parameter vector for specific gamma
     p_w=mvnpdf(w(i,:,j,n),[0 0 0 0],(gamma(i)*gamma(i))*I);%probablity of that particular gamma
     y=w(i,1,j,n)*(x(j))^3 + w(i,2,j,n)*(x(j))^2 + w(i,3,j,n)*(x(j))+ w(i,4,j,n)+v;%computation of y
     B=[ x(j)^3 ; x(j)^2 ;x(j);x(j)];
     Z= w(i,:,j,n)*B;%for computing MAP 
     prob_y_xw=mvnpdf(Z,sd^2);%Prior probablity
     W_map(i,:,j,n)=log(p_w)+log(prob_y_xw);%for estimation of map
     P(i,:,j,n)=[y,(x(j))^3,(x(j))^2,(x(j)),w(i,:,j,n),v,gamma(i),p_w,W_map(i)];%dataset
     
    end
end
end
%Estimating maximum probablity and computing L2estiamte
for n=1:10
for i=1:10
    [W_max(i),ind(i)]=max(W_map(:,:,i,n));
    W_true(i,:)=w(ind(i),:,i,n);
    for o=1:10
    L2_estimate(o,:,i,n)=abs((W_true(i,:)-w(o,:,i,n)).^2);
    sum_error_gamma(o,:,i,n)=sum(L2_estimate(o,:,i,n));
    end
end
end

%Data for Minimum,Maximum and Median
for i=1:10
  gamma_mi(i,:)=min(sum_error_gamma(i,:,:,:));
  gamma_ma(i,:)=max(sum_error_gamma(i,:,:,:));
  gamma_me(i,:)=median(sum_error_gamma(i,:,:,:));
end
for i=1:10
    gamma_min(i,:)=min(gamma_mi(i,:));
    gamma_max(i,:)=max(gamma_ma(i,:));
    gamma_med(i,:)=median(gamma_me(i,:));
end


figure(1)
subplot(3,1,1)
plot(gamma,gamma_min)
title('Plot of Squared Error Minimum in each Gamma')
xlabel('Gamma')
ylabel('Squared Error')
subplot(3,1,2)
plot(gamma,gamma_max)
title('Plot of Squared Error Maximum in each Gamma')
xlabel('Gamma')
ylabel('Squared Error')
subplot(3,1,3)
plot(gamma,gamma_med)
title('Plot of Squared Error Median in each Gamma')
xlabel('Gamma')
ylabel('Squared Error')
