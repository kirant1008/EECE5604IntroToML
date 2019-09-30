# EECE5664IntroToML
This folder contains all the Assignments Pertaining to EECE5664

Assignment 1:
      
      Question 2:
          Generate a plot of this log-likelihood-ratio function for the case a1 = 0,b1 = 1 and a2 =
          1,b2 = 2 using a suitable programming language.
      Solution:
          For this particular coding example we had to find the loglikelihood ratio , so by using the likelihood
          expression I got the required plot.
      
      Question 4:
          For the case μ = 1 and σ2 = 2 generate plots that visualize the class-conditional pdfs p(x|L =l)
          for l ∈ 1,2, as well as class posterior probabilities p(L = l|x) for l ∈ 1,2. Demonstrate the
          decision boundary you found earlier in these visualizations.
      Solution:
          Using the function normpdf I generated the required pdf's and used the decision boundary.I marked
          the decision boundary in both the cases.
          
      Question 5:
          Write code that takes in N, n, μ, and Σ and produces N samples of independent and identically
          distributed (iid) n-dimensional random vectors {x1,...,xN} drawn from N (μ,Σ) using the linear
          transformation technique applied to samples of z ∼ N (0,I).
      Solution:
          I set the values of N=4 and n=4.I generated random A and b using rand() function.As we get X using Z,
          I generated Random Z with help of mvnrnd() and put in equation Az+b.In this way I got $ different X1 X2
          X3 and X4.
          
      Code for all the above Questions are written in Matlab.
