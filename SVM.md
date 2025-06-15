Date- 30/4/25
Day-Wednesday

## Support Vector Machines
classification, regression and outlier detection

non-probabilistic binary linear classifier

1963: SVM algorithm was developed by **Vladimir N Vapnik** and **Alexey Ya. Chervonenkis**
1992: **Bernhard E. Boser**, **Isabelle M Guyon** and **Vladimir N Vapnik** suggested a way to create **non-linear classifiers** by applying the kernel trick to maximum-margin hyperplanes

**kernel trick**: implicitly map the inputs into high dimensional feature spaces. 

Separating Hyperplane

https://www.kaggle.com/code/prashant111/svm-classifier-tutorial
https://www.geeksforgeeks.org/support-vector-machine-algorithm/
https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/
https://scikit-learn.org/stable/modules/svm.html
https://www.youtube.com/@SerranoAcademy
https://www.youtube.com/watch?v=Lpr__X8zuE8

Calculus for machine learning - jason brownlee

Expanding Rate: increase slowly to increase margin slightly

random line
epoch : pick a large no 1000
learning rate
expanding rate

pick random point p, q

![[Pasted image 20250430221128.png]]
![[Pasted image 20250430221317.png]]

Polynomial Kernel
Radial Basis Function Kernel/ Gaussian Kernel 


### One Versus One Classification

k > 2,  
$$^kC_2$$
- suppose there are k>2 classes for a SVM problem
- (k, 2) SVMs comparing a pair of classes with each combination
- test observation is classified by tallying the assignments to each of the K classes

decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
ovo -> one versus one
### One Versus All Classification

A or non A - 135 obs
Rest of all

decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
ovr -> one versus rest

**Next Day**
Decision Tree


















