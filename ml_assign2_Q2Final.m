clc;
clear all;

% In this program we need to generate a data distribution
% with specified mean and covariance for each of the given case
% Prior probablities for all the three cases below is 0.5

%%%%%%%%%%%CASE 1%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 1
no_Samples=400;

c1_class_Mean1=[0; 0];

class_Cov1= eye(2);

c1_class_Mean2=[3 ;3];

class_Cov2= eye(2);

p_w=0.5;%prior probablity of class w1 and w2


c1_class_one=mvnrnd(c1_class_Mean1,class_Cov1,no_Samples*p_w);%Generation of Dataset using mvrnd for class1 

c1_class_two=mvnrnd(c1_class_Mean2,class_Cov2,no_Samples*p_w);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case1=[c1_class_one;c1_class_two];%DataSet for class 1

%Scatter Plot of Case 1 for generated dataset with class labels W1 and W2
figure(1)
subplot(1,2,1)
scatter(c1_class_one(:,1),c1_class_one(:,2))
hold on
scatter(c1_class_two(:,1),c1_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c1_inf_class1=zeros(1,2);
c1_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case1_inf_label=0;
for i=1:400
    pdf_classOne=mvnpdf(data_case1(i,:)',c1_class_Mean1,class_Cov1);
    pdf_classTwo=mvnpdf(data_case1(i,:)',c1_class_Mean2,class_Cov2);
    g=log(pdf_classOne)+log(p_w)-log(pdf_classTwo)-log(p_w);
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
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case1_og_class=[ones(no_Samples*p_w,1);2*ones(no_Samples*p_w,1)];

case1_noErrors= case1_og_class==case1_inf_label;
case1_countEr=0;
for j=1:400
    if(case1_noErrors(j)==0)
        case1_countEr=case1_countEr+1;
    end
end
p_error_case1=case1_countEr/no_Samples;    


fprintf('No. of Errors for case 1 are %i\n',case1_countEr)
fprintf('Prob. of Errors for case 1 is %i\n',p_error_case1)

%%%%%%%%%%%CASE
%%%%%%%%%%%2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 2
no_Samples=400;

c2_class_Mean1=[0; 0];

c2_class_Cov1= [3 ,1 ; 1 ,0.8];

c2_class_Mean2=[3 ;3];

c2_class_Cov2= [3 ,1 ; 1 ,0.8];

p_w=0.5;%prior probablity of class w1 and w2


c2_class_one=mvnrnd(c2_class_Mean1,c2_class_Cov1,no_Samples*p_w);%Generation of Dataset using mvrnd for class1 

c2_class_two=mvnrnd(c2_class_Mean2,c2_class_Cov2,no_Samples*p_w);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case2=[c2_class_one;c2_class_two];%DataSet for class 1

%Scatter Plot of Case 1 for generated dataset with class labels W1 and W2
figure(2)
subplot(1,2,1)
scatter(c2_class_one(:,1),c2_class_one(:,2))
hold on
scatter(c2_class_two(:,1),c2_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c2_inf_class1=zeros(1,2);
c2_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case2_inf_label=0;
for i=1:400
    c2_pdf_classOne=mvnpdf(data_case2(i,:)',c2_class_Mean1,c2_class_Cov1);
    c2_pdf_classTwo=mvnpdf(data_case2(i,:)',c2_class_Mean2,c2_class_Cov2);
    g2=log(c2_pdf_classOne)+log(p_w)-log(c2_pdf_classTwo)-log(p_w);
    
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g2>0
        c2_inf_class1=[c2_inf_class1;data_case2(i,:)];
        case2_inf_label=[case2_inf_label;1];
    else
        c2_inf_class2=[c2_inf_class2;data_case2(i,:)];
        case2_inf_label=[case2_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c2_inf_class1(1,:)=[];
c2_inf_class2(1,:)=[];
case2_inf_label(1)=[];

%Plot of Inferred Class labels W_1 and W_2 for case2
subplot(1,2,2)
scatter(c2_inf_class1(:,1),c2_inf_class1(:,2))
hold on
scatter(c2_inf_class2(:,1),c2_inf_class2(:,2))
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case2_og_class=[ones(no_Samples*p_w,1);2*ones(no_Samples*p_w,1)];

case2_noErrors= case2_og_class==case2_inf_label;
case2_countEr=0;
for j=1:400
    if(case2_noErrors(j)==0)
        case2_countEr=case2_countEr+1;
    end
end
p_error_case2=case2_countEr/no_Samples;    


fprintf('No. of Errors for case 2 are %i\n',case2_countEr)
fprintf('Prob. of Errors for case 2 is %i\n',p_error_case2)

%%%%%%%%%%%CASE 3%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 3
no_Samples=400;

c3_class_Mean1=[0; 0];

c3_class_Cov1= [2 ,0.5 ; 0.5 ,1];

c3_class_Mean2=[2; 2];

c3_class_Cov2= [2 ,-1.9 ; -1.9 ,5];

p_w=0.5;%prior probablity of class w1 and w2


c3_class_one=mvnrnd(c3_class_Mean1,c3_class_Cov1,no_Samples*p_w);%Generation of Dataset using mvrnd for class1 

c3_class_two=mvnrnd(c3_class_Mean2,c3_class_Cov2,no_Samples*p_w);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case3=[c3_class_one;c3_class_two];%DataSet for class 1
% mean_data_case_3=mean(data_case3);
% covar_data_case_3=cov(data_case3);

%Scatter Plot of Case 1 for generated dataset with class labels W1 and W2
figure(3)
subplot(1,2,1)
scatter(c3_class_one(:,1),c3_class_one(:,2))
hold on
scatter(c3_class_two(:,1),c3_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c3_inf_class1=zeros(1,2);
c3_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case3_inf_label=0;
for i=1:400
    c3_pdf_classOne=mvnpdf(data_case3(i,:)',c3_class_Mean1,c3_class_Cov1);
    c3_pdf_classTwo=mvnpdf(data_case3(i,:)',c3_class_Mean2,c3_class_Cov2);
    g3=log(c3_pdf_classOne)+log(p_w)-log(c3_pdf_classTwo)-log(p_w);
    
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g3>0
        c3_inf_class1=[c3_inf_class1;data_case3(i,:)];
        case3_inf_label=[case3_inf_label;1];
    else
        c3_inf_class2=[c3_inf_class2;data_case3(i,:)];
        case3_inf_label=[case3_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c3_inf_class1(1,:)=[];
c3_inf_class2(1,:)=[];
case3_inf_label(1)=[];
 
%Plot of Inferred Class labels W_1 and W_2 for case1 
subplot(1,2,2)
scatter(c3_inf_class1(:,1),c3_inf_class1(:,2))
hold on
scatter(c3_inf_class2(:,1),c3_inf_class2(:,2))
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case3_og_class=[ones(no_Samples*p_w,1);2*ones(no_Samples*p_w,1)];

case3_noErrors= case3_og_class==case3_inf_label;
case3_countEr=0;
for j=1:400
    if(case3_noErrors(j)==0)
        case3_countEr=case3_countEr+1;
    end
end
p_error_case3=case3_countEr/no_Samples;    


fprintf('No. of Errors for case 3 are %i\n',case3_countEr)
fprintf('Prob. of Errors for case 3 is %i\n',p_error_case3)


% In this program we need to generate a data distribution
% with specified mean and covariance for each of the given case
% Prior probablities for all the three cases below is 0.5

%%%%%%%%%%%CASE 4%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 4
no_Samples=400;

c4_class_Mean1=[0; 0];

class_Cov4= eye(2);

c4_class_Mean2=[3 ;3];

class_Cov4= eye(2);

p_w1=0.95;%prior probablity of class w1 and w2
p_w2=0.05;

c4_class_one=mvnrnd(c4_class_Mean1,class_Cov4,no_Samples*p_w1);%Generation of Dataset using mvrnd for class1 

c4_class_two=mvnrnd(c4_class_Mean2,class_Cov4,no_Samples*p_w2);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case4=[c4_class_one;c4_class_two];%DataSet for class 1

%Scatter Plot of Case 4 for generated dataset with class labels W1 and W2
figure(4)
subplot(1,2,1)
scatter(c4_class_one(:,1),c4_class_one(:,2))
hold on
scatter(c4_class_two(:,1),c4_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c4_inf_class1=zeros(1,2);
c4_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case4_inf_label=0;
for i=1:400
    pdf_classOne=mvnpdf(data_case4(i,:)',c4_class_Mean1,class_Cov4);
    pdf_classTwo=mvnpdf(data_case4(i,:)',c4_class_Mean2,class_Cov4);
    g=log(pdf_classOne)+log(p_w1)-log(pdf_classTwo)-log(p_w2);
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g>0
        c4_inf_class1=[c4_inf_class1;data_case4(i,:)];
        case4_inf_label=[case4_inf_label;1];
    else
        c4_inf_class2=[c4_inf_class2;data_case4(i,:)];
        case4_inf_label=[case4_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c4_inf_class1(1,:)=[];
c4_inf_class2(1,:)=[];
case4_inf_label(1)=[];

%Plot of Inferred Class labels W_1 and W_2 for case1 
subplot(1,2,2)
scatter(c4_inf_class1(:,1),c4_inf_class1(:,2))
hold on
scatter(c4_inf_class2(:,1),c4_inf_class2(:,2))
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case4_og_class=[ones(no_Samples*p_w1,1);2*ones(no_Samples*p_w2,1)];

case4_noErrors= case4_og_class==case4_inf_label;
case4_countEr=0;
for j=1:400
    if(case4_noErrors(j)==0)
        case4_countEr=case4_countEr+1;
    end
end
p_error_case4=case4_countEr/no_Samples;    


fprintf('No. of Errors for case 4 are %i\n',case4_countEr)
fprintf('Prob. of Errors for case 4 is %i\n',p_error_case4)

%%%%%%%%%%%CASE 5%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 5

c5_class_Mean1=[0; 0];

c5_class_Cov1= [3 ,1 ; 1 ,0.8];

c5_class_Mean2=[3 ;3];

c5_class_Cov2= [3 ,1 ; 1 ,0.8];


c5_class_one=mvnrnd(c5_class_Mean1,c5_class_Cov1,no_Samples*p_w1);%Generation of Dataset using mvrnd for class1 

c5_class_two=mvnrnd(c5_class_Mean2,c5_class_Cov2,no_Samples*p_w2);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case5=[c5_class_one;c5_class_two];%DataSet for class 1

%Scatter Plot of Case 5 for generated dataset with class labels W1 and W2
figure(5)
subplot(1,2,1)
scatter(c5_class_one(:,1),c5_class_one(:,2))
hold on
scatter(c5_class_two(:,1),c5_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c5_inf_class1=zeros(1,2);
c5_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case5_inf_label=0;
for i=1:400
    pdf_classOne=mvnpdf(data_case5(i,:)',c5_class_Mean1,c5_class_Cov1);
    pdf_classTwo=mvnpdf(data_case5(i,:)',c5_class_Mean2,c5_class_Cov1);
    g=log(pdf_classOne)+log(p_w1)-log(pdf_classTwo)-log(p_w2);
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g>0
        c5_inf_class1=[c5_inf_class1;data_case5(i,:)];
        case5_inf_label=[case5_inf_label;1];
    else
        c5_inf_class2=[c5_inf_class2;data_case5(i,:)];
        case5_inf_label=[case5_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c5_inf_class1(1,:)=[];
c5_inf_class2(1,:)=[];
case5_inf_label(1)=[];

%Plot of Inferred Class labels W_1 and W_2 for case1 
subplot(1,2,2)
scatter(c5_inf_class1(:,1),c5_inf_class1(:,2))
hold on
scatter(c5_inf_class2(:,1),c5_inf_class2(:,2))
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case5_og_class=[ones(no_Samples*p_w1,1);2*ones(no_Samples*p_w2,1)];

case5_noErrors= case5_og_class==case5_inf_label;
case5_countEr=0;
for j=1:400
    if(case5_noErrors(j)==0)
        case5_countEr=case5_countEr+1;
    end
end
p_error_case5=case5_countEr/no_Samples;    


fprintf('No. of Errors for case 5 are %i\n',case5_countEr)
fprintf('Prob. of Errors for case 5 is %i\n',p_error_case5)

%%%%%%%%%%%CASE 6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Variabels Required for case 6
no_Samples=400;

c6_class_Mean1=[0; 0];

c6_class_Cov1= [2 ,0.5 ; 0.5 ,1];

c6_class_Mean2=[2; 2];

c6_class_Cov2= [2 ,-1.9 ; -1.9 ,5];



c6_class_one=mvnrnd(c6_class_Mean1,c6_class_Cov1,no_Samples*p_w1);%Generation of Dataset using mvrnd for class1 

c6_class_two=mvnrnd(c6_class_Mean2,c6_class_Cov2,no_Samples*p_w2);%Generation of Dataset using mvrnd for class2

%Combining Dataset of class 1 and 2 in a single matrix
data_case6=[c6_class_one;c6_class_two];%DataSet for class 1
% mean_data_case_3=mean(data_case3);
% covar_data_case_3=cov(data_case3);

%Scatter Plot of Case 6 for generated dataset with class labels W1 and W2
figure(6)
subplot(1,2,1)
scatter(c6_class_one(:,1),c6_class_one(:,2))
hold on
scatter(c6_class_two(:,1),c6_class_two(:,2))
hold off
title('Plot of IID Samples With Class Label W_1 and W_2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%Variables to store inferred class labels point 
c6_inf_class1=zeros(1,2);
c6_inf_class2=zeros(1,2);


%Using Minimum Probablity of Error Classification Rule for generating
%labels
case6_inf_label=0;
for i=1:400
    pdf_classOne=mvnpdf(data_case6(i,:)',c6_class_Mean1,c6_class_Cov1);
    pdf_classTwo=mvnpdf(data_case6(i,:)',c6_class_Mean2,c6_class_Cov2);
    g=log(pdf_classOne)+log(p_w1)-log(pdf_classTwo)-log(p_w2);
    %g(x)=g1(x)+g2(x)
    %here,gi(x)=p(x/wi)*p(wi)
    %if g(x) is positive implies class 1 else class 2
    if g>0
        c6_inf_class1=[c6_inf_class1;data_case6(i,:)];
        case6_inf_label=[case6_inf_label;1];
    else
        c6_inf_class2=[c6_inf_class2;data_case6(i,:)];
        case6_inf_label=[case6_inf_label;2];
    end
end

%Deleting extra entries of [0,0]
c6_inf_class1(1,:)=[];
c6_inf_class2(1,:)=[];
case6_inf_label(1)=[];

%Plot of Inferred Class labels W_1 and W_2 for case1 
subplot(1,2,2)
scatter(c6_inf_class1(:,1),c6_inf_class1(:,2))
hold on
scatter(c6_inf_class2(:,1),c6_inf_class2(:,2))
hold off
title('Plot of Inferred Class Labels W_1 and W_2 using MAP')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2'},'Location','northeast')

%for generating No of error and Probablity of Error
case6_og_class=[ones(no_Samples*p_w1,1);2*ones(no_Samples*p_w2,1)];

case6_noErrors= case6_og_class==case6_inf_label;
case6_countEr=0;
for j=1:400
    if(case6_noErrors(j)==0)
        case6_countEr=case6_countEr+1;
    end
end
p_error_case6=case6_countEr/no_Samples;    


fprintf('No. of Errors for case 6 are %i\n',case6_countEr)
fprintf('Prob. of Errors for case 6 is %i\n',p_error_case6)
