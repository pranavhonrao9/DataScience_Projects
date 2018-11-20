
# coding: utf-8

# # Seattle Airbnb Open Data Analysis
# 

# # Content <font color= black>
The following Airbnb activity is included in this Seattle dataset: 
a) Listings: This csv file includes full descriptions and average review score 
b) Calendar: This csv file includes listing id and the price and availability for that day
# # Business Questions <font color= black>
a)How  room avaiablity varies through out the year in Seattle?
b)Mean prices analysis per month?
c)Price detection on the basis of available dataset?
# In[1]:


# Importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


get_ipython().run_line_magic('matplotlib', 'inline')


# # I. Data Exploration <font color= black>

# In[2]:


calendar_df = pd.read_csv('calendar.csv',thousands=',')


# In[3]:


calendar_df[:5]


# In[4]:


calendar_df.available.value_counts()


# In[5]:


calendar_df.isnull().sum()


# In[6]:


calendar_df = calendar_df.dropna(axis=0)


# In[7]:


calendar_df.info()


# In[8]:


calendar_df.describe()


# In[9]:


len(calendar_df.date.unique())


# In[10]:


calendar_df.head()


# ## II. Data Processing <font color= black>

# In[11]:


calendar_df.available[calendar_df.available == 't'] = 1
calendar_df.available[calendar_df.available == 'f'] = 0


# In[12]:


calendar_df['date'] = pd.to_datetime(calendar_df['date'])


# In[13]:


calendar_plot_df = pd.DataFrame(calendar_df.groupby(calendar_df['date'].dt.strftime('%B'))['available'].sum().sort_values())


# In[14]:


calendar_plot_df.reset_index(level=0, inplace=True)


# In[15]:


calendar_plot_df['available'].sum()


# In[16]:


calendar_plot_df


# In[17]:


calendar_plot_df.plot(x="date", y="available", kind="bar")


# What are the busiest times of the year to visit Seattle?
# 
# Ans: From the above graph, I can see January ,Feburary has less rooms available which make sense to me. As people would not prefer to put home on airbnb becuase of winter season. Also, in summer days Seattle being one of the famous cities to visit ,less rooms are avaiable. 
# 
# Interesingly , room availability do not vary that much beside March and December. I think ,in march people start putting there home on Airbnb listings but do not get that much of response causing high number of rooms avaiable. This is maybe because ,people travel less during that period. Surprisingly for December, room availability is high. 
# 
# 
# 

# In[18]:


calendar_df["price"]= calendar_df["price"].str.replace('$','')


# In[19]:


#price_removed_nan["price"] = price_removed_nan["price"].astype('float')
calendar_df["price"] = calendar_df["price"].str.replace(",","").astype(float)


# In[20]:


calendar_df.info()


# In[21]:


calendar_df['price'].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price in $')
plt.title('Histogram of listing prices')


# In[22]:


calendar_df['date'][:5]


# In[23]:


montlyprice_mean_df = calendar_df.groupby(calendar_df['date'].dt.strftime('%B'))['price'].mean().sort_values()


# In[24]:


montlyprice_mean_df.plot(x="date", y="price", kind="bar")

Mean prices analysis per month?

# In[25]:


# Model


# ## III. Data Analysis  <font color= black>

# In[26]:


listing_df = pd.read_csv('listings.csv',thousands=',')


# In[27]:


listing_df.info()


# In[28]:


listing_df.cancellation_policy[:5]


# In[29]:


df = listing_df[["host_response_rate", "host_acceptance_rate", "host_is_superhost",
               "host_listings_count", "property_type","room_type", "accommodates", "bathrooms", "bedrooms", 
               "beds","guests_included", "price", "review_scores_rating","review_scores_cleanliness","review_scores_location","review_scores_communication","review_scores_value","cancellation_policy", 
               "reviews_per_month","instant_bookable"]]


# In[30]:


df.isnull().sum()


# In[31]:


df =df.drop(columns='host_acceptance_rate')


# In[32]:


df.dtypes


# In[33]:


df.host_response_rate[:5]


# In[34]:


df.host_is_superhost[:5]


# In[35]:


df.cancellation_policy[:5]


# In[36]:


seattle_listing_df =df.dropna(axis=0)


# In[37]:


seattle_listing_df.info()


# In[38]:


seattle_listing_df.isnull().sum()


# In[39]:


seattle_listing_df['host_response_rate'] = seattle_listing_df['host_response_rate'].astype(str)
seattle_listing_df['price'] = seattle_listing_df['price'].astype(str)


# In[40]:


seattle_listing_df['host_response_rate'] = seattle_listing_df['host_response_rate'].str.replace("%", "").astype("float")
seattle_listing_df['price'] = seattle_listing_df['price'].str.replace("[$, ]", "").astype("float")


# In[41]:


seattle_listing_df.instant_bookable.unique()


# In[42]:


seattle_listing_df.host_is_superhost[seattle_listing_df.host_is_superhost == 't'] = 1
seattle_listing_df.host_is_superhost[seattle_listing_df.host_is_superhost == 'f'] = 0
seattle_listing_df['host_is_superhost']=seattle_listing_df['host_is_superhost'].astype(int)


# In[43]:


seattle_listing_df.instant_bookable[seattle_listing_df.instant_bookable == 't'] = 1
seattle_listing_df.instant_bookable[seattle_listing_df.instant_bookable == 'f'] = 0
seattle_listing_df['instant_bookable']=seattle_listing_df['instant_bookable'].astype(int)


# In[44]:


seattle_listing_df.info()


# In[45]:


object_dtype = seattle_listing_df.select_dtypes(include=['object']).columns
seattle_listing_df[object_dtype].head()


# In[46]:


seattle_listing_df.property_type.unique()


# In[47]:


seattle_listing_df.room_type.unique()


# In[48]:


seattle_listing_df.cancellation_policy.unique()


# In[49]:


dummy_var_list = pd.get_dummies(seattle_listing_df[object_dtype])


# In[50]:


dummy_var_list.head()


# In[51]:


seattle_listing_df= seattle_listing_df.drop(object_dtype,axis=1)
seattle_listing_df = pd.merge(seattle_listing_df,dummy_var_list, left_index=True, right_index=True)
seattle_listing_df.head()


# In[52]:


seattle_listing_df.info()


# ## IV. Feature Visualization  <font color= black>

# In[53]:


seattle_listing_df['price'].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price in $')
plt.title('Histogram of listing prices')


# In[54]:


from matplotlib.pyplot import *
plt.plot( 'price', 'bedrooms', data=seattle_listing_df, marker='o', color='mediumvioletred')
plt.ylabel('bedrooms')
plt.xlabel('Listing price in $')


# In[55]:


seattle_listing_df.pivot(columns = 'bedrooms',values = 'price').plot.hist(stacked = True,bins=50)
plt.xlabel('Listing price in $')


# 
# ## IV. Modeling <font color= black>

# In[56]:


X = seattle_listing_df.drop('price', axis =1)
y = seattle_listing_df['price']


# In[57]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=1)
seattle_linear_model = linear_model.LinearRegression()
seattle_linear_model.fit(X_train, y_train)


# In[58]:


y_train_pred = seattle_linear_model.predict(X_train)
y_test_pred = seattle_linear_model.predict(X_test)


# In[59]:


rms_ols2= math.sqrt(mean_squared_error(y_test,y_test_pred))
print('MSE train: %.4f, test: %.4f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.4f, test: %.4f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[60]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=300, 
                               criterion='mse', 
                               random_state=4)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.4f, test: %.4f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.4f, test: %.4f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

