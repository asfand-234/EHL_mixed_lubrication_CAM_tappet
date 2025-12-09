Read the text file "symbolic_data.txt". It is data having first two columns are  independent variables (S, F).
And third column is dependent variable (E).

TASK:

Your task is to create a perfect Gaussian model for E as function of S and F.
Try to start from this base line model. And keep adding necessary mathematical functions and optimizing constants to get accuracy.
E = c1* exp(- ((S- c2)^2)/(c3*(w1)^2) - ((F- c4)^2)/(c5*(w2)^2))

where c1,c2,c3,c4,c5 are constants to be determined.
And w1 ,w2 are constants called gaussian widths to be determined.

where w is a constant gaussian width.
TESTING CRITERIA:
1) You must find error between each value of E. And error must be less than 20% for each value of E. 
Must check and verify each value of predicted E before commiting, else keep optimizing and improving function
2)Do not generate radial basis functions (RBF). Try the baseline above model and keep optiming its constants, mathematical functioms etc.
3) Try to make the gaussian model simple and with less constants. To achieve reduce the accuracy from 100% to 80%. But not below 80% for each value of E.
4) E must not be input variable in formula. As E is dependent variable.
5) Once target achieved then provide precisely the final model in commit with neccessary explanations.
