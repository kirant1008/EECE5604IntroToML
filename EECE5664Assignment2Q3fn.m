clc;
clear;
close all;

% In this program we need to generate a data distribution
% with specified mean and covariance for each of the given case

error1=lda([0; 0],eye(2),[3 ;3],eye(2),0.5,0.5,1);

error2=lda([0; 0],[3 ,1 ; 1 ,0.8],[3 ;3],[3 ,1 ; 1 ,0.8],0.5,0.5,2);

error3=lda([0; 0],[2 ,0.5 ; 0.5 ,1],[2 ;2],[2 ,-1.9 ; -1.9 ,5],0.5,0.5,3);

error4=lda([0; 0],eye(2),[3 ;3],eye(2),0.95,0.05,4);

error5=lda([0; 0],[3 ,1 ; 1 ,0.8],[3 ;3],[3 ,1 ; 1 ,0.8],0.95,0.05,5);

error6=lda([0; 0],[2 ,0.5 ; 0.5 ,1],[2 ;2],[2 ,-1.9 ; -1.9 ,5],0.95,0.05,6);


function p_error = lda(mean1,cov1,mean2,cov2,pw1,pw2,m)
%Defining required variables
no_Samples=400;

c1_class_Mean1=mean1;

class_Cov1= cov1;

c1_class_Mean2=mean2;

class_Cov2= cov2;

p_w1=pw1;%prior probablity of class w1 and w2
p_w2=pw2;

c1_class_one=mvnrnd(c1_class_Mean1,class_Cov1,no_Samples*p_w1);%Generation of Dataset using mvrnd for class1 

c1_class_two=mvnrnd(c1_class_Mean2,class_Cov2,no_Samples*p_w2);%Generation of Dataset using mvrnd for class2


%generating within and between matrix
within_case1=class_Cov1+class_Cov2;
between_case1=(c1_class_Mean1-c1_class_Mean2)*(c1_class_Mean1-c1_class_Mean2)';


%getting the dominant eigan values
[V,D]= eig(inv(within_case1)*(between_case1));
x= diag(D);
%inorder to find position of dominant eigen value vector in x
if x(1)>x(2)
    w=V(:,1);
else
    w=V(:,2);
end

%Getting projection of class 1 and class 2 
z1= transpose(w'*c1_class_one');
z2= transpose(w'*c1_class_two');

%Combing two sets projection class 1 and class 2
z=[z1;z2];

%Projecting threshold on direction vector
t=w'*((c1_class_Mean1+c1_class_Mean2)./2);

%Linear Discriminant classifier of generated threshold 
y1=0;
y2=0;

inf_label=0;
for i=1:400
    
    if z(i,1)<t
        y1=[y1,z(i,1)];
        inf_label=[inf_label;1];
    else
        y2=[y2,z(i,1)];
        inf_label=[inf_label;2];
    end
end

y1(1)=[];
y2(1)=[];
inf_label(1)=[];

a=size(y1);
b=size(y2);

%counting no of errors in classification
og_class=[ones(no_Samples*p_w1,1);2*ones(no_Samples*p_w2,1)];

noErrors= og_class==inf_label;
countEr=0;
for j=1:400
    if(noErrors(j)==0)
        countEr=countEr+1;
    end
end
p_error=(countEr/no_Samples)*100;

%Plotting Linear Discriminant Scores
figure(m)

subplot(2,1,1)

scatter(c1_class_one(:,1),c1_class_one(:,2))
hold on
scatter(c1_class_two(:,1),c1_class_two(:,2))
axis equal
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

subplot(2,1,2)
plot(y1(1,:),zeros(1,a(2)),'r*');
hold on;
plot(y2(1,:),zeros(1,b(2)),'bo');
axis equal
xline(t)
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('y-Projection Vector')
ylabel('1D LDA Scores')
legend({'W_1','W_2'},'Location','northeast')
end


