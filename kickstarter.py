#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:42:44 2018

@author: abinavrameshsundararaman
"""
import plotly
plotly.__version__
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from dfply import *
import matplotlib.pyplot as plt
from pylab import *
from ggplot import *

kickstarter=pd.read_excel("Dataset.xlsx")
kickstarter_actual=kickstarter


'''
#################################################################### Clean data #################
'''

# Creating a modified column for launch_to_state_change_days
for i in range(len(kickstarter)):
    kickstarter.loc[i,'launch_to_state_change_days_modified'] = ((datetime.datetime.strptime(kickstarter.loc[i,'state_changed_at'], '%Y-%m-%dT%H:%M:%S'))-datetime.datetime.strptime(kickstarter.loc[i,'launched_at'], '%Y-%m-%dT%H:%M:%S')).days


kickstarter=kickstarter.loc[:, ~kickstarter.columns.isin(['name','deadline','state_changed_at','created_at','launched_at','name_len','blurb_len','deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday','launch_to_state_change_days'])]

kickstarter=kickstarter.dropna()


sc = datetime.datetime.strptime(kickstarter_actual.state_changed_at[1], '%Y-%m-%dT%H:%M:%S')
l = datetime.datetime.strptime(kickstarter_actual.launched_at[1], '%Y-%m-%dT%H:%M:%S')

kickstarter.columns

kickstarter['usd_goal'] = kickstarter['goal']*kickstarter['static_usd_rate']



kickstarter_actual.isna().sum()

# dummify variable

state_dummies=pd.get_dummies(kickstarter.state,prefix='state')
disable_communication_dummies = pd.get_dummies(kickstarter.disable_communication,prefix='disable_communication')

kickstarter = kickstarter.assign(country_categorical = [a if ((a=='US') | (a =='GB')| (a =='CA')) else 'Other' for a in kickstarter['country']])

kickstarter = kickstarter.assign(seasons = ['Summer' if ((a>=6) & (a <=8)) else ('Spring' if ((a>=3) & (a <=5)) else 'Winter') for a in kickstarter['launched_at_month']])

country_dummies = pd.get_dummies(kickstarter.country_categorical,prefix='country')
staff_pick_dummies = pd.get_dummies(kickstarter.staff_pick,prefix='staff_pick')


category_dummies = pd.get_dummies(kickstarter.category,prefix='category')
spotlight_dummies = pd.get_dummies(kickstarter.spotlight,prefix='spotlight')
currency_dummies = pd.get_dummies(kickstarter.currency,prefix='currency')

kickstarter.columns

# Remove other columsn that are already dummified

kickstarter_categorical=kickstarter.loc[:, ~kickstarter.columns.isin(['country','disable_communication','staff_pick','category','spotlight','currency','country_categorical','pledged','goal','seasons'])]

kickstarter=kickstarter.loc[:, ~kickstarter.columns.isin(['country','state','disable_communication','staff_pick','category','spotlight','currency','country_categorical','pledged','goal','seasons'])]


cleaned_data=pd.concat([kickstarter, state_dummies,disable_communication_dummies,country_dummies,staff_pick_dummies,category_dummies,spotlight_dummies,currency_dummies], axis=1)

cleaned_data_categorical =pd.concat([kickstarter_categorical, disable_communication_dummies,country_dummies,staff_pick_dummies,category_dummies,spotlight_dummies,currency_dummies], axis=1)

cleaned_data_categorical.columns
cleaned_data.columns

kickstarter=kickstarter.loc[:, ~kickstarter.columns.isin(['launch_to_state_change_days'])]
cleaned_data_categorical = cleaned_data_categorical.loc[:, ~cleaned_data_categorical.columns.isin(['launch_to_state_change_days'])]
cleaned_data = cleaned_data.loc[:, ~cleaned_data.columns.isin(['launch_to_state_change_days','seasons'])]

#'launch_to_state_change_days' has a lot of NAs
# 'usd_pledged' is very correlated with 'pledged'

cleaned_data=cleaned_data.dropna()

cleaned_data.columns
cleaned_data_categorical.columns


'''
#################################################### Which are the most important variables #############
'''
# For Regression-- Pledged


y_regression=cleaned_data.loc[:, 'usd_pledged']

X_regression=cleaned_data.loc[:, ~cleaned_data.columns.isin (['usd_pledged','project_id'])]

randomforest = RandomForestRegressor(random_state=0)

model_regression = randomforest.fit(X_regression, y_regression)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model_regression, threshold=0.001)
sfm.fit(X_regression, y_regression)
for feature_list_index in sfm.get_support(indices=True):
    print(X_regression.columns[feature_list_index])
    
pd.DataFrame(list(zip(X_regression.columns,model_regression.feature_importances_)), columns = ['predictor','Gini_coefficient']).sort_values('Gini_coefficient',ascending = False)



######### For Classification

## For Classification --
cleaned_data_categorical.columns
cleaned_data_categorical.isna().sum()


cleaned_data_categorical=cleaned_data_categorical.loc[:, ~cleaned_data_categorical.columns.isin(['launch_to_state_change_days'])]

cleaned_data_categorical=cleaned_data_categorical.dropna()
cleaned_data_categorical.state.unique()

# create_to_launch_days,currency_SEK, launched_at_month
# subsetting only for successful and failed
cleaned_data_categorical = cleaned_data_categorical[(cleaned_data_categorical.state =="failed")|(cleaned_data_categorical.state =="successful")]

y=cleaned_data_categorical.loc[:, 'state']

X=cleaned_data_categorical.loc[:, ~cleaned_data_categorical.columns.isin (['state','project_id','state_category'])]

randomforest_classifier = RandomForestClassifier(random_state=0)

model_classification = randomforest_classifier.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model_classification, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
pd.DataFrame(list(zip(X.columns,model_classification.feature_importances_)), columns = ['predictor','Gini_coefficient']).sort_values('Gini_coefficient',ascending = False)
'''
###############################################################  MODELLING  ########################
'''
################################### For Regression 

# Build a  Linear Regression model

####### Linear regression code
y_regression=cleaned_data.loc[:, 'usd_pledged']

X_regression=cleaned_data.loc[:,['usd_goal','create_to_launch_days','blurb_len_clean','category_Sound','deadline_month','deadline_day','staff_pick_False','staff_pick_True','category_Web','category_Hardware','country_US','country_GB','country_Other','country_CA','launch_to_deadline_days','name_len_clean','created_at_yr','deadline_yr']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size = 0.33, random_state = 5)
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_fit = standardizer.fit(X_train)
X_train_std=X_train_fit.transform(X_train)
X_test_std = X_train_fit.transform(X_test)

#### Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train_std, y_train)

y_test_pred = model.predict(X_test_std)
from sklearn.metrics import mean_squared_error
mse_linear = mean_squared_error(y_test, y_test_pred)
print(mse_linear)


####### Random forest regressor code

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 5,n_estimators=100,max_depth = 5)
model_regressor_rf = rf.fit(X_train,y_train)
y_pred_regression_rf = model_regressor_rf.predict(X_test)
mse = mean_squared_error(y_pred_regression_rf, y_test)
print(mse)

####### SVM code

from sklearn.svm import SVR
svr = SVR(kernel='rbf', epsilon=0.01)
model_regression_svr = svr.fit(X_train_std,y_train)
y_pred_regression_svr= model_regression_svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_regression_svr)
print(mse)

# Run decision tree
from sklearn.tree import DecisionTreeRegressor
decisiontree = DecisionTreeRegressor()
model_regression_decisiontree = decisiontree.fit(X_train, y_train)
y_regression_predict_decisiontree=model_regression_decisiontree.predict(X_test)
mse = mean_squared_error(y_test, y_regression_predict_decisiontree)
print(mse)

########################################## For Classification ##########################

########### Run logistic regression
y_cat=cleaned_data_categorical.loc[:, 'state']

#X_cat=cleaned_data_categorical.loc[:, ~cleaned_data_categorical.columns.isin (['state','project_id','state_category'])]

X_cat=cleaned_data_categorical.loc[:,['usd_goal','create_to_launch_days','blurb_len_clean','category_Sound','deadline_month','deadline_day','staff_pick_False','staff_pick_True','category_Web','category_Hardware','country_US','country_GB','country_Other','country_CA','launch_to_deadline_days','name_len_clean','created_at_yr','deadline_yr','category_Plays']]
X_cat=cleaned_data_categorical.loc[:,['usd_goal','create_to_launch_days','blurb_len_clean','category_Sound','deadline_month','deadline_day','staff_pick_False','staff_pick_True','category_Web','category_Hardware','country_US','country_GB','country_Other','country_CA','launch_to_deadline_days','name_len_clean','created_at_yr','deadline_yr']]

cleaned_data_categorical.columns
#X_cat=cleaned_data_categorical.loc[:, ['spotlight_False']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cat, y_cat, test_size = 0.5, random_state = 5)
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_train_fit = standardizer.fit(X_train)
X_train_std=X_train_fit.transform(X_train)
X_test_std = X_train_fit.transform(X_test)


### Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=5)
model_lr = lr.fit(X_train,y_train)
y_predict_lr=model_lr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict_lr)

# Use Random forest

cleaned_data_categorical.columns
cleaned_data_categorical=cleaned_data_categorical.dropna()
randomforest_classifier = RandomForestClassifier(random_state = 5,n_estimators = 100,max_features=4,max_depth = 5)
model_classification = randomforest_classifier.fit(X_train, y_train)
y_classification_predict=model_classification.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_classification_predict))


# Run decision tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state = 5, max_features = 4,max_depth = 5)
model_classification_decisiontree = decisiontree.fit(X_train, y_train)
y_classification_predict_decisiontree=model_classification_decisiontree.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_classification_predict_decisiontree)


'''
########################################## For Clustering ##########################

'''
cleaned_data.columns

X_temp = kickstarter_actual 
X_temp.isna().sum()
X_temp = X_temp.loc[:, ~X_temp.columns.isin(['launch_to_state_change_days'])]
X_temp=X_temp.dropna()
X_temp['usd_goal'] = X_temp['goal']*X_temp['static_usd_rate']



X_temp=pd.concat([X_temp,pd.get_dummies(X_temp.spotlight,prefix='spotlight'),pd.get_dummies(X_temp.staff_pick,prefix='staff_pick'),pd.get_dummies(X_temp.state,prefix='state'),pd.get_dummies(X_temp.country,prefix='country'),pd.get_dummies(X_temp.category,prefix='category')], axis=1)

X_temp.columns


X=X_temp.loc[:,['launched_at_yr', 'country_US','usd_goal']]

X.isna().sum()

# Using K-Means

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)



# Check Silhoutte scores
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X_std)
labels = model.predict(X_std)
silhouette_score(X_std,labels)


from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std,labels)


df = pd.DataFrame({'label':labels,'silhouette':silhouette})  #dataframe has label of each obeservation and silohette score of each observation 

print('Average Silhouette Score for Cluster 0: ',np.average(df[df['label'] == 0].silhouette)) 
print('Average Silhouette Score for Cluster 1: ',np.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',np.average(df[df['label'] == 3].silhouette)) 


## CONCAT 
X.loc[:,'cluster'] = labels


##SUBSET
dataset_0 = X[(X.cluster == 0)]
dataset_1 = X[(X.cluster == 1)]
dataset_2 = X[(X.cluster == 2)]
dataset_3 = X[(X.cluster == 3)]
dataset_4 = X[(X.cluster == 4)]


# Plot cluster membership, this is a scatter plot, comment it in inplac of dendogram to see 
from matplotlib import pyplot
plt.scatter(X['usd_goal'],X['launched_at_yr'],c= labels, cmap='rainbow')

X_temp_temp=X_temp.merge(X,how='right')

X_temp_temp=pd.concat([X_temp,X.cluster], axis=1)


a=X_temp_temp[X_temp_temp.cluster==2].project_id

X_temp_temp.to_csv("cluster_best_v2.csv")


'''
############################ Summary Statistics--Preliminary data analysis ###################
'''
## Check why Currency_SEK is always featuring
# country 'SE'-- for currency ='SEK' -- FInal decision : Dont include Currency SEK into the picture


pd.crosstab(kickstarter_actual.state, kickstarter_actual.country, rownames=['state'], colnames=['country'])
pd.crosstab(kickstarter_actual.state, kickstarter_actual.country, rownames=['state'], colnames=['currency'])
pd.crosstab(y_test, y_classification_predict, rownames=['state'], colnames=['currency'])
pd.crosstab(cleaned_data_categorical.state, columns='count')

### Plot created_at vs pledged to see if its of some use


### Crosstab for spotlight vs successful
pd.crosstab(kickstarter_actual.state, kickstarter_actual.spotlight, rownames=['State'], colnames=['Spotlight'])


### Crosstab for staffpick vs successful

kickstarter_actual[['staff_pick', 'state']].pivot_table(columns='state', index='staff_pick', aggfunc=len).plot(kind='bar', figsize=(8, 8))
plt.title(' Number of projects')
plt.xlabel("Staffpick")
plt.ylabel("Number of projects")
plt.show()

### Crosstab for country vs state
kickstarter_actual = kickstarter_actual.assign(country_categorical = [a if ((a=='US') | (a =='GB')| (a =='CA')) else 'Other' for a in kickstarter_actual['country']])

kickstarter_actual[['country_categorical', 'state']].pivot_table(columns='state', index='country_categorical', aggfunc=len).plot(kind='bar', figsize=(8, 8))
plt.title(' Number of projects')
plt.xlabel("Staffpick")
plt.ylabel("Number of projects")
plt.show()

### Country

kickstarter_actual['country'].value_counts().plot(kind='bar', color='red')
plt.title('Number of Kickstarter Projects Per Country')
plt.xlabel('Country')
plt.ylabel('Number of Kickstarter Projects ')


#Distribution of Categories that have failed
kickstarter_actual[kickstarter_actual.state =="failed"]['category'].value_counts().plot(kind='bar', color='red')
plt.title('Frequency of Kickstarter Categories of Failed Projects')
plt.xlabel('Project Categories')
plt.ylabel('Project Frequency ')


kickstarter_actual[kickstarter_actual.state =="successful"]['category'].value_counts().plot(kind='bar', color='green')
plt.title('Number of Kickstarter Projects ')
plt.xlabel('Project Categories')
plt.ylabel('Project Frequency ')
plt.legend(( 'Success'))

temp2=kickstarter_actual[(kickstarter_actual.state =="successful")|(kickstarter_actual.state =="failed")]
temp2.state
temp2=temp2.loc[:, ~temp2.columns.isin(['launch_to_state_change_days'])]
temp2.isna().sum()
temp2=temp2.dropna()

#### Staff pick vs success or failure
kickstarter_actual.columns
pd.crosstab(kickstarter_actual.staff_pick, kickstarter_actual.state, rownames=['Staff Pick'], colnames=['State'])
pd.crosstab(kickstarter_actual.category, kickstarter_actual.state, rownames=['Staff Pick'], colnames=['State'])

###### TO see if category sound has more pledged or not
import matplotlib.pyplot as plt
temp=pd.DataFrame(kickstarter_actual.groupby('category').pledged.mean())
temp.reset_index(level=0, inplace=True)

bar1=plt.bar(temp.category,temp.pledged,color = "red" )
bar1[8].set_color('blue')
bar1[18].set_color('blue')
bar1[21].set_color('blue')
plt.xticks(rotation=90)
plt.title("Average Pledged Amount per Project Category")
plt.xlabel("Categories")
plt.ylabel("Average Pledged Amount")

###### TO see if Country has more pledged or not
import matplotlib.pyplot as plt
temp=pd.DataFrame(kickstarter_actual.groupby('country').usd_pledged.mean())
temp.reset_index(level=0, inplace=True)

bar1=plt.bar(temp.country,temp.usd_pledged,color = "red" )
plt.xticks(rotation=90)
plt.title("Average Pledged Amount per Country")
plt.xlabel("Country")
plt.ylabel("Average Pledged Amount")


# blurb_len_clean vs avg pledged amount

plt.scatter(kickstarter_actual.blurb_len_clean,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount across different Blurb Lengths")
plt.xlabel("Blurb Length")
plt.ylabel("Pledged Amount")

temp=kickstarter_actual.loc[:,['blurb_len_clean','state']]
temp = temp[(temp.state=='failed')|(temp.state=='successful')]
plt.scatter(temp.blurb_len_clean,temp.state,color = "red" )
plt.title("Blurb Length")
plt.xlabel("Pledged Amount")
plt.ylabel("Pledged Amount vs Blurb Length")


# name_len_clean vs avg pledged amount

plt.scatter(kickstarter_actual.name_len_clean,kickstarter_actual.pledged,color = "green" )
plt.title("Pledged Amount across different Name Lengths ")
plt.xlabel("Name Length")
plt.ylabel("Pledged Amount")

temp=kickstarter_actual.loc[:,['blurb_len_clean','state']]
temp = temp[(temp.state=='failed')|(temp.state=='successful')]
plt.scatter(temp.blurb_len_clean,temp.state,color = "red" )
plt.title("Blurb Length")
plt.xlabel("Pledged Amount")
plt.ylabel("Pledged Amount vs Blurb Length")

# create_to_launch_days vs avg pledged amount
kickstarter_actual.columns
plt.scatter(kickstarter_actual.create_to_launch_days,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount  vs Create to Launch Days")
plt.xlabel("Create to Launch Days")
plt.ylabel("Pledged Amount")


# create_to_launch_days vs avg pledged amount
kickstarter_actual.columns
plt.scatter(kickstarter_actual.deadline_day,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount  vs Deadline Days")
plt.xlabel("Deadline Days")
plt.ylabel("Pledged Amount")

plt.scatter(kickstarter_actual.deadline_month,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount  vs Deadline Month")
plt.xlabel("Deadline Month")
plt.ylabel("Pledged Amount")

plt.scatter(kickstarter_actual.deadline_yr,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount  vs Deadline Year")
plt.xlabel("Deadline Year")
plt.ylabel("Pledged Amount")

plt.scatter(kickstarter_actual.deadline,kickstarter_actual.pledged,color = "red" )
plt.title("Pledged Amount  vs Deadline Year")
plt.xlabel("Deadline Year")
plt.ylabel("Pledged Amount")

# created_at_day vs average pledged amount
import matplotlib.pyplot as plt
temp=pd.DataFrame(kickstarter_actual.groupby('created_at_weekday').pledged.mean())
temp.reset_index(level=0, inplace=True)

bar1=plt.bar(temp.created_at_weekday,temp.pledged,color = "red" )
plt.autoscale(enable=True, axis='x', tight=True)
plt.title("Average Pledged Amount per Weekday")
plt.xlabel("Project Created Day")
plt.ylabel("Average Pledged Amount")

# Which category has highest Success rate
temp= pd.DataFrame(kickstarter_actual.groupby(['category','state'])['category','state'].count()/kickstarter_actual.groupby(['category'])['category','state'].count())

