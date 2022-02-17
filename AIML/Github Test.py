#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from sklearn import (datasets, neighbors,
                     model_selection as skms,
                     linear_model, metrics)


# load 2019 College Football dataset and extract desired information
cfb_orig_df = pd.read_csv("CFB2019.csv")
# extract bowl eligibility from the Wins-Loss column
# first separate Win-Loss into 2D array of Win and Loss strings
w_l = np.array([w_l_str.split('-') for w_l_str in cfb_orig_df['Win-Loss']])
# extract first column as wins (and convert to integers)
wins = w_l[:,0].astype(int)

# extract some of the existing columns as a starting point for our smaller dataset
cfb_df = cfb_orig_df[['Rushing Yards per Game',
                      'Rush Yards Per Game Allowed',
                      'Pass Yards Per Game',
                      'Pass Yards Per Game Allowed']].copy()
# add the wins column
cfb_df['Wins'] = wins
# make the team column the index labels
cfb_df.index = cfb_orig_df['Team']

# remove Iowa so that not used for training/testing 
# (as we will perform a separate prediction later)
iowa_values = cfb_df.loc['Iowa (Big Ten)',:]
print(iowa_values)
cfb_df.drop('Iowa (Big Ten)',inplace=True)

# display first five rows
display(cfb_df.head())

# separate features from targets
# obtain target as the wins column
cfb_target = cfb_df['Wins']
# obtain features as the DataFrame with the wins column dropped
cfb_features = cfb_df.drop(['Wins'],axis=1)


# separate into train and test sets (30% of data for testing)
(cfb_train_ftrs, cfb_test_ftrs,
 cfb_train_tgt, cfb_test_tgt) = skms.train_test_split(cfb_features,
                                                      cfb_target,
                                                      test_size=.30)

# set up k-NN (k=3) regression model
model = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = model.fit(cfb_train_ftrs, cfb_train_tgt)

# apply trained model to test data and evaluate using rmse
predictions = fit.predict(cfb_test_ftrs)
score = np.sqrt(metrics.mean_squared_error(cfb_test_tgt,
                                           predictions))
print(f'Model RMSE: {score:0.2f}')

# feature values defined below for your convenience (but you may
# obtain them in a different way if you prefer)
ia_rush_per_game = iowa_values['Rushing Yards per Game']
ia_rush_per_game_allowed = iowa_values['Rush Yards Per Game Allowed']
ia_pass_per_game = iowa_values['Pass Yards Per Game']
ia_pass_per_game_allowed = iowa_values['Pass Yards Per Game Allowed']
ia_wins = iowa_values['Wins']

# YOUR CODE HERE PartII A
#raise NotImplementedError()
ia_df = cfb_train_ftrs.copy()

ia_distances = np.sqrt(((cfb_features['Rushing Yards per Game'] - ia_rush_per_game)**2)
                 + (cfb_features['Rush Yards Per Game Allowed'] - ia_rush_per_game_allowed)**2
                 + (cfb_features['Pass Yards Per Game'] - ia_pass_per_game)**2
                 + (cfb_features['Pass Yards Per Game Allowed'] - ia_pass_per_game_allowed)**2
)

print('dist = ', ia_distances)

ia_df['Distance'] = ia_distances
ia_df['Wins'] = cfb_train_tgt

display(ia_df)

ia_sorted = ia_df.sort_values('Distance')
top_three_df = ia_sorted.iloc[0:3,:]
display(top_three_df)

mean_three = np.mean(top_three_df['Wins'])
print(mean_three)

# Make an Iowa ftrs dataframe
# This simplifies the use of:  fit.predict()
iowa_ftrs_df = pd.DataFrame([iowa_values])
iowa_ftrs_df = iowa_ftrs_df.drop("Wins",axis=1)
print("iowa_ftrs_df:")
display(iowa_ftrs_df)


# YOUR CODE HERE PartII B
#raise NotImplementedError()

predictions = fit.predict(iowa_ftrs_df)
print(predictions)


# YOUR OPTIONAL CODE HERE
predictions = fit.predict(cfb_test_ftrs)

bowl_eligible_actual = cfb_test_tgt >= 6
bowl_eligible_predicted = predictions >= 6

score2 = metrics.accuracy_score(bowl_eligible_actual, bowl_eligible_predicted)

print('accuracy of bowl predictions: ', score2)
inverse_dist = 1/top_three_df['Distance']
weights = inverse_dist/inverse_dist.sum()
weighted_mean = np.dot(top_three_df['Wins'], weights)

print('predicted wins weighted =', weighted_mean)

# same way with kNN Regressor

model_weighted = neighbors.KNeighborsRegressor(n_neighbors=3,weights='distance')
fit_weighted = model_weighted.fit(cfb_train_ftrs, cfb_train_tgt)
predictions_weighted = fit_weighted.predict(iowa_ftrs_df)

#print the predictions_weighted

print("Weighted predictions: ", predictions_weighted)
