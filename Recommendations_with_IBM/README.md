# Recommendations with IBM
### Description:
In this project a dataset from the IBM Watson studio was provided. And the goal was to create a recommendation engine based on this. This recommendation engine should point users to new articles that they might like and keep them engaged on the platform.
There current implementation just shows the newest article, but with a recommendation engine it might be the most interesting for a given user.

### Requirements:

* numpy==1.15.1
* pandas==0.23.4

### Details:
The project was divided into several subjections:
I. Exploratory Data Analysis

This first step was to explore the available dataset. To better understand the available data and to move on to the next steps.

II. Rank Based Recommendations

The first Recommendation system, which was implemented was a Rank Based system. Here we find the most popular articles simply based on the most interactions. It is similar to the what's hot on youtube or other platforms.

III. User-User Based Collaborative Filtering

In this step the recommendation was based on similarity. So rather then simply recommending the articles, which have the most interaction, instead the individual user gets taken into account. This way he gets results based on what similar users like.

IV. Content Based Recommendations (EXTRA - NOT REQUIRED)

This part was skipped for now. But I might come back to this later in time.

V. Matrix Factorization

Finally, you will complete a machine learning approach to building recommendations. Using the user-item interactions, you will build out a matrix decomposition. This way we can get an idea of how well we can predict new articles an individual might interact with. And in the end a short discussion can be found on how to move on from here.



