clc;
clear;
close all;

%Values of Dataset
no_Samples=10000;
mean(:,1)=[-1;0]; cov(:,:,1)=0.1*[10 -4;-4,5]; c1_pw=0.15;%for class 1 
mean(:,2)=[1;0] ; cov(:,:,2)=0.1*[5 0;0,2];c2_pw=0.35 ;%for class 2 
mean(:,3)=[0;1] ; cov(:,:,3)= 0.1*eye(2); c3_pw=0.5 ;%for class 3


class_Priors=[c1_pw,c2_pw,c3_pw];
prior_threshold=[0,cumsum(class_Priors)];%inorder to generate datasets
prob_uni=rand(1,no_Samples);%generating 10000 samples of probablities
og_Labels=zeros(1,no_Samples);


figure(1)
%generation of dataset
for i=1:3
    pntr=find(prob_uni>=prior_threshold(i) &  prob_uni<=prior_threshold(i+1));
    og_Labels(1,pntr)=i*ones(1,length(pntr));
    count_samples(1,i)=length(pntr);
    data(:,pntr)=mvnrnd(mean(:,i),cov(:,:,i),length(pntr))';
    figure(1)
    subplot(2,1,1)
    plot(data(1,pntr),data(2,pntr),'.'); axis equal, hold on,
end
hold off
title('Plot of IID Samples With Class Label W_1 W_2 and W_3')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2','W_3'},'Location','northeast')

%Variables to store inferred class labels point 
inf_class=zeros(1,no_Samples);

%Using Minimum Probablity of Error Classification Rule for generating
%labels
countEr=0;
correct_class1=[0;0];
correct_class2=[0;0];
correct_class3=[0;0];
incorrect_class=[0;0];
%Data of misclassification as class 1 class 2 and class 3
miss_class1=[0;0];
miss_class2=[0;0];
miss_class3=[0;0];

for i=1:no_Samples
    pdf_classOne=(mvnpdf(data(:,i),mean(:,1),cov(:,:,1)))*c1_pw;
    pdf_classTwo=(mvnpdf(data(:,i),mean(:,2),cov(:,:,2)))*c2_pw;
    pdf_classThree=(mvnpdf(data(:,i),mean(:,3),cov(:,:,3)))*c3_pw;
    g1=log(pdf_classOne)+log(c1_pw)-log(pdf_classTwo)-log(c2_pw)-log(pdf_classThree)-log(c3_pw);
    g2=log(pdf_classTwo)+log(c2_pw)-log(pdf_classOne)-log(c1_pw)-log(pdf_classThree)-log(c3_pw);
    g3=log(pdf_classThree)+log(c3_pw)-log(pdf_classOne)-log(c1_pw)-log(pdf_classTwo)-log(c2_pw);
    
   if pdf_classOne>pdf_classTwo & pdf_classOne>pdf_classThree
        inf_class(1,i)=1;
    elseif pdf_classTwo>pdf_classOne & pdf_classTwo>pdf_classThree
        inf_class(1,i)=2;
    else 
        inf_class(1,i)=3;
    end
    
    noErrors(1,i)= og_Labels(1,i)==inf_class(1,i);
    
    if og_Labels(1,i)==inf_class(1,i)
       if og_Labels(1,i)==1 
            correct_class1=[correct_class1,data(:,i)];
       elseif og_Labels(1,i)==2
            correct_class2=[correct_class2,data(:,i)];
       elseif og_Labels(1,i)==3
            correct_class3=[correct_class3,data(:,i)];
       end
    else
        incorrect_class=[incorrect_class,data(:,i)];
        if inf_class(1,i)==1
             miss_class1=[miss_class1,data(:,i)];
        elseif inf_class(1,i)==2
             miss_class2=[miss_class2,data(:,i)];
        elseif inf_class(1,i)==3
             miss_class3=[miss_class3,data(:,i)];
        end
    end
    
    countEr=length(incorrect_class);%Counting no. of errors.
end
correct_class1(:,1)=[];
correct_class2(:,1)=[];
correct_class3(:,1)=[];
incorrect_class(:,1)=[];
miss_class1(:,1)=[];
miss_class2(:,1)=[];
miss_class3(:,1)=[];



subplot(2,1,2)
plot(correct_class1(1,:),correct_class1(2,:),'.'); axis equal; hold on;
plot(correct_class2(1,:),correct_class2(2,:),'.'); axis equal; hold on;
plot(correct_class3(1,:),correct_class3(2,:),'.'); axis equal; hold on;
% plot(incorrect_class(1,:),incorrect_class(2,:),'*'); axis equal; hold on;
hold off
title('Plot of Inferred Samples With Class Label W_1 W_2 W_3')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'W_1','W_2','W_3','Missclassifications'},'Location','northeast')

p_Error=(countEr/no_Samples)*100;
display(p_Error,'The Probablity of Error is: ')

display(count_samples,'Number of Samples in each class are:')

display(countEr,'Number of Samples missclassified:')

C = confusionmat(inf_class,og_Labels);
figure(2)
confusionchart(C);

figure(3)

subplot(3,1,1)
plot(miss_class1(1,:),miss_class1(2,:),'.','color','r'); axis equal;
title('Plot of Missclassification in Class 1')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'Miss W_1'},'Location','northeast')

subplot(3,1,2)
plot(miss_class2(1,:),miss_class2(2,:),'.','color','b'); axis equal;
title('Plot of Missclassification in Class 2')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'Miss W_2'},'Location','northeast')

subplot(3,1,3)
plot(miss_class3(1,:),miss_class3(2,:),'.','color','g'); axis equal;
title('Plot of Missclassification in Class 3')
xlabel('Feature Value x_1')
ylabel('Feature Value x_2')
legend({'Miss W_3'},'Location','northeast')

