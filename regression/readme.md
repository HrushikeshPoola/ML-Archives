# Logistic Regression

Dataset : [Penguin Dataset](./datasets/penguins.csv)

### 1. Primary Objective
In this part of the assignment, we will implement a logistic regression model from scratch. We work on the penguin dataset which pertains to particular penguin species and their features. We use the sigmoid function to obtain the probability for prediction while fitting the dataset. This function maps a real number to a value between 0 and 1. The logistic regression model predicts the target value for the text data, in this case, sex has been chosen.

### 2. Data Exploration
The penguin dataset has been imported from the data file and as a part of the preprocessing pipeline we have identified features that have missing values and omitted them. Secondly, we encoded the features which were in string format into numerical values using the pandas library which provides us with the pd.dummies function enabling the creation of columns that have the corresponding string values. Particularly the year column has been dropped as it didn’t intuitively seem to be contributing a lot to the dataset. Later a statistical map of the overall dataset has been generated to get a blanket overview of it.

### 3. Normalization, Target Selection, Splitting
A simplistic normalization as mentioned in the problem statement where we calculate the scaled value by using the maximum and minimum value of each feature value has been performed on the dataset.
We have used the sex as the target variable as it seemed convenient given that it is a binary value by default in the current dataset available.

### 4. Behaviour of Loss Function
We can deduce that the loss has been decreasing for multiple values of the learning rates as the iterations go on increasing. Also, the value is in terms of the highest learning rate parameter as it should be for each of the individual plot curves, Consider the green curve and the orange one given that the alpha for the latter is a lot higher it falls steeper earlier than the above one as it should.

### 5. Hyperparamters
Here we primarily have decided to tweak the existing parameters as a part of the imple- mentation of logistic regression namely Number of iterations and Learning Rate. Here are three different setups with different learning rates and the number of iterations, and how they affect the performance of a logistic regression model:
- Setup 1: Learning rate = 0.001, Number of iterations = 1000 In this setup, the learning rate is small and the number of iterations is low. This means that the algorithm will take small steps in the direction of the gradient and will not iterate many times to update the weights. The model may converge too slowly or may not converge at all. As it turns the accuracy results - 0.5074
 
- Setup 2: Learning rate = 0.025, Number of iterations = 1000 In this setup, the learning rate is substantial and the number of iterations is low here as well. This means that the algorithm will take faster steps in the direction of the gradient and will not iterate a large number of times but improves a lot more over the last model. As it turns the accuracy results - 0.805
- Setup 3: Learning rate = 0.01, Number of iterations = 10000 In this setup, the learning rate is lesser than in the previous instance but the number of iterations is very high. This means that the algorithm will take slower steps in the direction of the gradient but will iterate a large number of times. As it turns the accuracy results - 0.8805

### 6. Observations
- As we have seen in the above instances the higher the number of iterations always improved the accuracy and dominated the learning rate emphasis on the overall accuracy.
- The graphs clearly indicated that there is a distinct drop in the loss curve with a higher learning rate which saves a lot more time to obtain the most efficient model.
- We have also tweaked the initialization of weights and starting with zeros has given better observation in comparison to starting with ones or uniform random values. It could be that it was more related to the fact the data size was small and
the weight variation is not necessarily conclusive enough to make such a claim.

### 7. Logistic Regression Benefits and Drawbacks
- Logistic Regression models offer a clear understanding of the impact of indepen- dent variables on the outcome variable, which makes them easy to interpret. The coefficients of the model indicate the magnitude of the change in the dependent variable that occurs as a result of a unit change in the independent variable.
- Logistic Regression models are less prone to overfitting than other classification algorithms, such as neural networks. Logistic Regression models are particularly useful when the outcome variable is binary, i.e., it has only two possible values.
- When using Logistic Regression models, it is assumed that there is a linear re- lationship between the independent variables and the dependent variable. If this assumption is violated, the model may produce erroneous outcomes.
- Logistic Regression models may be affected by outliers, which can distort the model parameters and lead to inaccurate predictions. Additionally, The ability of Logistic Regression models to model decision boundaries is limited to linear rela- tionships, which can be disadvantageous when the decision boundary is nonlinear.

