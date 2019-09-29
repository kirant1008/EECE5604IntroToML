clc;
clear all;
%All the constants such as Mean,Prior Probablity,Standard deviations.
mu1=0;
probL1=1/2;
probL2=1/2;
sig1=1;
mu2=1;
sig2=2;

%Generating feature Vector x ranging from -10 to 10.
x1=linspace(-10,10);

%Generating Class Conditional Probablities
px_L1= normpdf(x1,mu1,sig1);
px_L2= normpdf(x1,mu2,sig2);

%Generating plot
figure(1)
plot(x1,px_L1)
hold on
plot(x1,px_L2)
hold off
%Drawing a threshold at theta=1.11
xline(1.11,':g','LineWidth',2)
title("Plot of Class Conditional PDF's P(x/L=l)")
xlabel("Feature value 'x'")
ylabel("Class Conditional PDF's")
legend({'p(x/L=1)','p(x/L=2)','Boundary Theta=1.11'},'Location','northeast')

figure(2)

%Inorder to find Posterior Probabilties We use conditional pdfs and
%Evidence p(x)

px=(px_L1.*(probL1))+(px_L2.*(probL2));
plx1=(px_L1.*(probL1))./px;
plx2=(px_L2.*(probL2))./px;

%Plotting Posterior Probablities with Boundary at Theta=0.5
plot(x1,plx1)
hold on
plot(x1,plx2)
hold off

%Drawing a threshold at theta=0.5
yline(0.5,':g','LineWidth',2)
title("Plot of Class Posterior PDF's P(L=l/x)")
xlabel("Feature value 'x'")
ylabel("Posterior PDF's")
legend({'p(L=1/x)','p(L=2/x)','Boundary Theta=0.5'},'Location','northeast')
