# Churn case Study
![Banner](images/map.png)

# Table of contents

* [EDA](#EDA)
* [Modeling](#Modeling)
* [Results](#Results)
* [Conclusion](#Conclusion)
* [Caveats](#Caveats)
* [Next Steps](#Next-Steps)


# Background

The data, a collection of users who signed up in January 2014, was pulled on July 1, 2014. We considered a user retained if they were “active” (i.e. took a trip) in the preceding 30 days (from the day the data was pulled). In other words, a user is "active" if they have taken a trip since June 1, 2014.

A presentation including the following points:

* What model did you use in the end? Why?
* Alternative models you considered? Why are they not good enough?
* What performance metric did you use to evaluate the model? Why?
* Based on insights from the model, what plans do you propose to reduce churn?
* What are the potential impacts of implementing these plans or decisions? What performance metrics did you use to evaluate these decisions, why?

# Plan

* what is my y variable
* what is "active"
* think about process..
* a pipeline..
* try diff models and try diff things..
* grid search
* keep an eye out for leakage - if cols are na when you are trying to predict

1. Perform any cleaning, exploratory analysis, and/or visualizations to use the provided data for this analysis.
2. Build a predictive model to help determine the probability that a rider will be retained.
3. Evaluate the model. Focus on metrics that are important for your statistical model.
4. Identify / interpret features that are the most influential in affecting your predictions.
4. Discuss the validity of your model. Issues such as leakage. For more on leakage, see this essay on Kaggle, and this paper: Leakage in Data Mining: Formulation, Detection, and Avoidance.
4. Repeat 2 - 5 until you have a satisfactory model.
4. Consider business decisions that your model may indicate are appropriate. Evaluate possible decisions with metrics 4. that are appropriate for decision rules.

hypotheses we had:
1. if rating goes down, maybe will churn
2. if didn't take a trip in february (within first month), maybe they will churn
3. if taking luxury trips maybe not churning
4. people that didn't rate at all more likely to churn

# EDA

* How did we compute the target? 
    * To create the target column we isolated the users that had taken a trip since June 1, 2014 and labeled them as 'active.'
    * Identified 14,595 active users out of 40,000
* Converted phone type and city to dummy variables
* 319 NaN values identified for phone os
* Dropped rows with NaN values for column 'avg_rating_by_driver' because it represented only a small subset of the data
* Ran T-test to determine the effect of NaN values in 'avg_rating_of_driver' on dataset
    * This helps us see if this subset has a different distribution than the rest of the dataset
    * Used average distance column as our comparator
    * t-statistic = -16.601760294457094
    * p-value = 7.626466059905339e-61
    * Given these results, we can reject the null hypothesis. The calculated p-value indicates that our NaN values do have an effect on our overall dataset and therefore decided not to drop these rows but rather replace our NaNs with the mean
* columns created:
    * 'active' - Boolean; True indicates user is active, False indicates user is not active
    * 'rating_stated' - Boolean; True indicates user has avg rating of driver, False indicates user has NaN for avg rating of driver


![Scatter_Matrix](images/)
![correlation_matrix](images/correlationmatrix.png)
![heat_map](images/)


# Modeling
* Models Explored:
    * Linear Classifier
![placeholder](images/)
    * Decision Tree (Classification)
![placeholder](images/)
    * Random Forest (Classification)
![placeholder](images/)

Feature Importances:

# Results

# Conclusion

# Caveats

# Next Steps

