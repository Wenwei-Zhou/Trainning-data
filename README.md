# Trainning-data
Based on the training data, create three supervised machine learning (ML) models for predicting actual_productivity.

# Report
Report
Read garments_worker_productivity.csv file, use dataframe.info(verbose=True) to 
print information of all columns select the columns, and set the features and target 
for train and test the model. Remove the object data columns in features because 
training and testing datasets cannot directly handle string data and most machine 
learning algorithms expect numerical input, but also can replace the string data with 
number by LabelEncoder(). Used np.isnan to find and replace the missing value NaN 
in features columns with value 0. Implement train_test_split function and 
normalization to split the dataset into training set and test set, 70 percent into 
training set and 30 percent into testing set. Create MinMaxScaler check all feature 
values lie within the specified range.

The KNN regression algorithm predicts the output value by taking the average or 
weighted average of the output values of the K nearest neighbours. The average or 
weighted average of the output values of a new data point’s K closest neighbour is 
taken into account by the algorithm when determining its output value. In contrast to 
classification tasks, where the output value is a discrete class label, this task produces 
a continuous variable, such as a real number. The hyperparameters in GridSearchCV 
function are param_grid, n_jobs, scoring, error_score. The KNN algorithm's 
n_neighbors hyperparameter indicates how many neighbours are used for 
prediction. The range is set to 1 to 21 with a step size of 2 in the provided code. As a 
result, the grid search will assess the model's effectiveness for a range of n_neighbor 
values, including 1, 3, 5, 7,..., 21. Weight hyperparameter uniform' assigns equal 
weight to each neighbor, while 'distance' assigns weights proportional to the inverse 
of their distance. Metric hyperparameter use 'euclidean' uses the Euclidean distance, 
'manhattan' uses the Manhattan distance, and 'minkowski' is a generalization of the 
previous two metrics. CV parameter RepeatedStratifiedKFold cross-validation with 10 
splits and 3 repeats is employed. The output of performance on test set is 
0.020401235994148736, a lower MSE indicates better performance. The output of
performance on train set is 0.015677360911351947, the output of performance on 
train set is lower than the output of performance on test set, it indicates the model is 
overfitting.

When dealing with non-linear correlations between the input variables and the 
target variable, SVR is especially helpful. It is efficient in identifying intricate patterns 
in the data and can handle both linear and non-linear regression issues. The 
parameter kernel 'linear' kernel corresponds to linear regression, while the 'rbf' 
kernel allows for non-linear regression by mapping the data into a higherdimensional 
space. C is evaluated for 0.0001, 1, and 10 values. More regularisation is possible 
with a lower C value than with a higher C value, which has the opposite effect. For 
each C value, the grid search will assess how well the model performs. Gamma is 
taken into consideration for the numbers 1, 10, and 100. Smoother decision borders 
are produced by a wider Gaussian kernel, which is indicated by a smaller gamma 
value, whereas more localised decision boundaries are produced by a bigger gamma 
value. The output 0.020401235994148736 is looked like off by the square root, the 
output also displays the message Fitting 5 folds for each of 18 candidates, totalling 90 
fits, this means the model uses 5 folds cross-validation for train and evaluate, the 
hyperparameter search is evaluating 18 different combinations of hyperparameters, 
and totalling 90 fits is 5 folds and 18 candidates’ combination. The output of
performance on train set is 0.012384285885946027, the output of performance on 
train set is lower than the output of performance on test set, it indicates the model is 
overfitting.

In decision tree each internal node represents a decision based on a feature, and 
each leaf node represents a class label or a predicted value. It divides the data 
depending on feature values. In randomforest, by averaging the predictions of 
different decision trees, Random Forest, in comparison to a single decision tree, 
increases predictive power and decreases overfitting. Choosing a random subset of 
features at each split, also adds unpredictability, aiding in decorrelation of the trees 
and boosting variety. Decision tree parameter max depth restricts the number of 
splits and controls the complexity of the tree. It is varied from 2 to 9, incrementing by 
1 in each step. Randomforest hyperparameters n_estimators representing the 
number of decision tree will take in the randomforest and it use for the prediction, 
max depth restricts the depth of each tree, controlling the complexity and potential 
overfitting. The output of performance on test set is 0.01649823541122549 and the 
output of performance on train set is 0.0056328234433409416, the output of
performance on train set is lower than the output of performance on test set, it 
indicates the model is overfitting.

Feature importance helps identify the most influential features and provides insights 
into the underlying relationships between the features and the target variable. 
permutation importance is calculated using the permutation_importance function, 
and the resulting importance scores are assigned to the importance_score variable. 
Use the KNN best model as trained model, test feature X_test, target variable y_test, 
number of permutation repetitions n_repeats, and the random state. In this 
permutation importance with using KNN best model as trained model is display 
actual_productivity has the highest importance score, which mean the model heavily 
depends on this feature for predictions. SVR higher importance scores are smv and 
no_of_workers indicating features that have a larger impact on the model's 
performance. Decisiontree higher importance score is targeted_productivity
indicating features that have a larger impact on the model's performance. Random 
forest higher importance score is targeted_productivity indicating features that have 
a larger impact on the model's performance.
