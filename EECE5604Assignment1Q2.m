clc;
clear all;

%all the defined values of a1 b1 a2 b2
a1=0;
a2=1;
b1=1;
b2=2;

%Generating Feature Value x with linspace from -100 to 100
x1=linspace(-100,100);

% Using Reduced form of loglikelihood Ratio
lx = log(b2/b1)+ abs((x1-a2)/b2)-abs((x1-a1)/b1)

%Ploting Log-likelihood ratio
figure(1)
plot(x1,lx)
title('Plot of Log-likelihood-ratio Function')
xlabel('Feature value "x"')
ylabel('Log-likelihodd ratio "l(x)"')
legend({'Log-likelihodd ratio "l(x)"'},'Location','northeast')
