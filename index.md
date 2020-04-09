---
layout: default
---

# House Price Predictions with Machine Learning

### Introduction

Predicting how much a house is worth is a common problem today. For people selling their house, it is important to know how much their property can be sold for - how can they maximize their profit while having a reasonable chance of actually selling? For people buying a house, it is important to know whether a house is overpriced, reasonable, or listed at a bargain.

One approach to predicting house prices is using machine learning. Machine learning is the construction and use of computer programs that “learn” patterns and interactions from the data we give them. In particular, we will use supervised machine learning which means we learn from data where we already have the answers - we have the house prices of the houses in our dataset. 


### The Data

To address this problem, we chose to use the [Kaggle House Pricing Dataset](https://www.kaggle.com/c/home-data-for-ml-course/overview), which includes variables such as SalePrice (the property’s sale price in dollars - these are our ‘answers’), different variables representing the size and shape of the property, what types and how many rooms the building has, whether it has road access, the neighbourhood, type of building, location with respect to main roads and/or railroads, various aspects of the materials used to build the house, the year built and/or remodelled and the current condition. Here is a sneak peek at the first few rows of this dataset:

![image](https://media.github.students.cs.ubc.ca/user/7101/files/1f641580-79d1-11ea-84b1-3a2453ac1781)

Additionally, here is a quick summary of our data including important information, such as missing cells (entries that have been left blank), and how many data points we have: 

![Screen Shot 2020-04-07 at 5 36 04 PM](https://media.github.students.cs.ubc.ca/user/7101/files/3efb3e00-79d1-11ea-8f77-e76fdb6cf1e9)

Another important step to understanding the data you are handling is thinking about outliers. Outliers are data points that lie far outside the common ranges seen in your dataset. For example, if you have a set of ten people and their heights, nine of which are within 5’4” and 6’1”, and one of which is 7’0”, the 7-foot tall person is an outlier. 

To assess this in our dataset, we first looked at the frequency distribution of SalePrice. This shows us what sale prices are common and which are not. For example, we can see that prices between $100,000 and $300,000 are quite common whereas prices over $400,000 are very uncommon:

Though these high values are quite uncommon, they are not separated from the rest of the distribution - for this reason, we will not treat them as outliers for this analysis. 

Another way to explore outliers is to look at the relationship between an explanatory variable (a variable that contributes to our prediction in some way) and the response variable (the thing we are trying to predict - in our case, SalePrice). For instance, looking at GrLivArea (square footage of above-ground living area) versus SalePrice:


We can see there are some outliers to the far right that do not seem to follow the general trend We can see there are some outliers to the far right that do not seem to follow the general trend of the rest of the data. Is this a problem though? How much does this actually influence our prediction? To assess the impact of these outliers, we can simply calculate what the mean and variance of the GrLivArea data are with and without the outliers. It turns out, with the outliers, the mean of GrLivArea is about 1515.4637 sq.ft., and the variance is about 525.4804. Without the outliers, the mean is again about 1502.7688 sq.ft., and the variance is about 494.5385. We can see that the outliers do not influence these values very much so we do not need to be concerned about them. 

Another important step that will prepare us for the training of machine learning models is preprocessing our data. Preprocessing entails reformatting categorical variables, scaling numerical variables, and handling missing values. Without going into too much detail, our preprocessing involved all of these aspects. This is an important step to ensure our model can handle and pull as much information as possible from our data. 

### The Model

Now our data is finally ready to be fed into some machine learning models. There are many different types of models that have already been created to handle various tasks using machine learning. They each perform different calculations to ‘learn’ from the data and make predictions. But how do we know which model to use? Which model is best suited for our data/problem? A good first step here is to try scikit-learn’s [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). This is a very simple model that, by default, always predicts the mean of the values of the response variable in the training set. Note that this makes no use of the relationship any of the explanatory variables have with the response variable. Therefore, the DummyRegressor is not used as a solution for real problems, but rather as a baseline to compare with other more complex regressors. For most scikit-learn models, there is a score function that, for regressors, gives us the R^2 score. Without going into too much detail, the closer this score is to 1, the better the model. For our dataset, DummyRegressor yielded a score of -0.02267, which is very close to 0.

One more complex model to try is scikit-learn’s [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). This model fits a set of decision trees on subsets of factors (explanatory variables) from our training data, and averages the results from these trees for a prediction. An overview of decision trees, in particular scikit-learn’s implementation of them, can be found [here](https://scikit-learn.org/stable/modules/tree.html). In this case, we are using decision trees for regression. Using this model gives us a score of 0.86192. Clearly, this performs much better than the DummyRegressor model we tried first, so it is a viable solution to our problem. But we can do better. 

Note that in the previous section, we saw that our dataset includes some outliers. Scikit-learn’s [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html?highlight=huberregressor#sklearn.linear_model.HuberRegressor) is robust to outliers. This means that the outliers in our dataset do not influence the results of the model fitting very much, but they are also not ignored entirely. On our dataset, this model resulted in a score of 0.88816. This is slightly better than our RandomForestRegressor.

Another model to try is scikit-learn’s [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) model. This regression model optimizes the linear least squares function. For more information on linear least squares, see [this wikipedia page](https://en.wikipedia.org/wiki/Linear_least_squares), or do some googling of your own! This model gives a score of 0.89753 on our dataset - even better than HuberRegressor. 

Finally, [CatBoost](https://catboost.ai/docs/concepts/about.html) is an algorithm like advanced, fancy decision trees. It can work with diverse data types and can often be used for searches, recommendation systems, personal assistants, self-driving cars, weather prediction, etc. It also aims to provide great results with default parameters, which helps us save time tuning parameters. For our problem, this model gives a score of 0.8994. 

CatBoostRegressor gives us a very good score and this will probably be the best model to use in our case. However, one last step we can take to try to improve our performance is hyperparameter optimization. Hyperparameters are parameters that we give the model when we first create it. Each model takes different hyperparameters and has defaults to use if we do not pass in our own parameters. These parameters can affect a variety of aspects of the model. For instance, with random forests, an important hyperparameter is how many trees to use. This determines the complexity of the model, and can affect how well our model performs on training and test data. In our analysis, we tried using hyperparameter tuning to make our RIdge score better, but any improvement was minimal - sometimes the default hyperparameters are just good!

### Conclusion
Overall, for this dataset, the CatboostRegressor performed the best, based on the R^2 score. It is important to note that there are other ways to measure the performance of your model, and that it may not perform as well on new data as your training and testing data. Specifically, the dataset we used in this analysis only spans sales from 2006 to 2010, so the results we get from learning from this dataset may not be applicable now, 10 years later, due to inflation and long-term trends in the housing market. Additionally, there may be other time-related trends in this dataset that we have ignored in this analysis, such as seasonal dependencies in the sales of houses. These could have important contributions to our predictions as well. 

### Sources
“Decision Trees.” Scikit-Learn, scikit-learn.org/stable/modules/tree.html.

“Linear Least Squares.” Wikipedia, Wikimedia Foundation, 21 Feb. 2020, en.wikipedia.org/wiki/Linear_least_squares.

"Overview of CatBoost." CatBoost, catboost.ai/docs/concepts/about.html.

“User Guide.” Scikit-Learn, scikit-learn.org/stable/user_guide.html.
