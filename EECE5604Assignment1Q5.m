clc;
clear all;

%Setting Values of N and n for A and b
N=4
n=4

%for loop for generating N samples of iid n-dimensional random vectors
for c=1:N

   Xn=zeros(N)
   A1 = rand(n) %Generating a Random n*n Matrix
   b=rand(n,1)  %Randomly generating a random vector b
   I=eye(n)
   z0=zeros(n,1)
   R =transpose(mvnrnd(z0,I,1)) %Generating a random vector from normal 
                                %Ditribution with mean 0 and Co-Variance=I
   x= A1*R + b                  %Using linear transformation
   Xn(:,c)=x                    %Concatinating the X1,X2,X3,X4
   
end


