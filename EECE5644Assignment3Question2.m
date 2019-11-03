clc;
clear ;
close all;

[lda_error , lda_count] = lda([0; 0],[2 ,0.5 ; 0.5 ,1],[2 ;2],[2 ,-1.9 ; -1.9 ,5],0.3,0.7);
[map_error , map_count] = map([0; 0],[2 ,0.5 ; 0.5 ,1],[2 ;2],[2 ,-1.9 ; -1.9 ,5],0.3,0.7);
function [p_errormap,case1_countEr] = map(mean1,cov1,mean2,cov2,pw1,pw2)
no_Samples=1000;

c1_class_Mean1=mean1;

class_Cov1= cov1;

c1_class_Mean2=mean2;

class_Cov2= cov2;

p_w1=pw1;%prior probablity of class w1 and w2
p_w2=pw2;

c1_class_one=mvnrnd(c1_class_Mean1,class_Cov1,no_Samples*p_w1);%Generation of Dataset using mvrnd for class1 

c1_class_two=mvnrnd(c1_class_Mean2,class_Cov2,no_Samples*p_w2);%Generation of Dataset using mvrnd for class2
ent_data=[c1_class_one;c1_class_two];%combination of two dataset
%Combining Dataset of class 1 and 2 in a single matrix
data_case1=[c1_class_one;c1_class_two];%DataSet for class 1

%Scatter Plot of Case 1 for generated dataset with class labels W1 and W2
figure(1)
subplot(1,2,1)
scatter(c1_class_one(:,1),c1_class_one(:,2))
hold on
scatter(c1_class_two(:,1),c1_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+'},'Location','northeast')

%Variables to store inferred class labels point 
c1_inf_class1=zeros(1,2);
c1_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case1_inf_label=0;
for i=1:no_Samples
    pdf_classOne=mvnpdf(data_case1(i,:)',c1_class_Mean1,class_Cov1);
    pdf_classTwo=mvnpdf(data_case1(i,:)',c1_class_Mean2,class_Cov2);
    g=log(pdf_classOne)+log(p_w1)-log(pdf_classTwo)-log(p_w2);
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g>0
        c1_inf_class1=[c1_inf_class1;data_case1(i,:)];
        case1_inf_label=[case1_inf_label;1];
    else
        c1_inf_class2=[c1_inf_class2;data_case1(i,:)];
        case1_inf_label=[case1_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c1_inf_class1(1,:)=[];
c1_inf_class2(1,:)=[];
case1_inf_label(1)=[];

%Plot of Inferred Class labels W_1 and W_2 for case1 
subplot(1,2,2)
scatter(c1_inf_class1(:,1),c1_inf_class1(:,2))
hold on
scatter(c1_inf_class2(:,1),c1_inf_class2(:,2))
hold on

%for generating No of error and Probablity of Error
case1_og_class=[ones(no_Samples*p_w1,1);2*ones(no_Samples*p_w2,1)];

case1_noErrors= case1_og_class==case1_inf_label;
case1_countEr=0;
for j=1:no_Samples
    if(case1_noErrors(j)==0)
        case1_countEr=case1_countEr+1;
        scatter(ent_data(j,1),ent_data(j,2),'x','k');
    end
end
hold off
title('Plot of IID Samples after MAP classification With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+','missclassified'},'Location','northeast')
p_errormap=case1_countEr/no_Samples*100;    

end




function [p_errorlda,countEr] = lda(mean1,cov1,mean2,cov2,pw1,pw2)
%Defining required variables
no_Samples=1000;

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
for i=1:no_Samples
    
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
for j=1:no_Samples
    if(noErrors(j)==0)
        countEr=countEr+1;
    end
end
p_errorlda=(countEr/no_Samples)*100;

%Plotting Linear Discriminant Scores
figure(2)

subplot(2,1,1)

scatter(c1_class_one(:,1),c1_class_one(:,2))
hold on
scatter(c1_class_two(:,1),c1_class_two(:,2))
axis equal
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'q_-','q_+'},'Location','northeast')

subplot(2,1,2)
plot(y1(1,:),zeros(1,a(2)),'r*');
hold on;
plot(y2(1,:),zeros(1,b(2)),'bo');
axis equal
xline(t)
title('Plot of IID Samples With Class Label q_- and q_+')
xlabel('y-Projection Vector')
ylabel('1D LDA Scores')
legend({'q_-','q_+','b'},'Location','northeast')
end