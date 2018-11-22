# Seattle_Airbnb_DataScienceProject
https://www.kaggle.com/airbnb/seattle/home

# Context

Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. As part of the Airbnb Inside initiative, this dataset describes the listing activity of homestays in Seattle, WA.

# Content

The following Airbnb activity is included in this Seattle dataset:

a)Listings, including full descriptions and average review score 

b)Reviews, including unique id for each reviewer and detailed comments 

c)Calendar, including listing id and the price and availability for that day

# Business Questions

a)How  room avaiablity varies through out the year in Seattle?

b)Mean prices analysis per month?

c)Price detection on the basis of available dataset?


# Data Sources

https://www.kaggle.com/airbnb/seattle

# Libraries

http://www.numpy.org/
https://pandas.pydata.org/
https://matplotlib.org/
https://scikit-learn.org/stable/



# Analysis
After the data exploration and analysis , following is the analysis I was able derive:

a)How  room avaiablity varies through out the year in Seattle?
![image](https://user-images.githubusercontent.com/8425221/48756454-40ef2080-ec67-11e8-97d2-8e50971c951d.png)
From the above graph, I can see January ,Feburary has less rooms available which make sense to me. As people would not prefer to put home on airbnb becuase of winter season. Also, in summer days Seattle being one of the famous cities to visit ,less rooms are avaiable.

Interesingly , room availability do not vary that much beside March and December. I think ,in march people start putting there home on Airbnb listings but do not get that much of response causing high number of rooms avaiable. This is maybe because ,people travel less during that period. Surprisingly for December, room availability is high.


b)Mean prices analysis per month?

![image](https://user-images.githubusercontent.com/8425221/48756822-9d9f0b00-ec68-11e8-9ef2-cf67202ed274.png)

For the prices , average mean prices are less in the month of Jan and Feb which is fair as inclement weather can affect people choices to out. On contractory , summer time is most favorable time in the year where people prefer to go outside locations.Hence , we can see in significient  spike in prices in summer duration.




c)Price variance with the number of bedrooms:

![image](https://user-images.githubusercontent.com/8425221/48756772-69c3e580-ec68-11e8-816c-66c6fd775273.png)

As expected , number of bedrooms is one of the prime feature to determine the price of the house. More the number of bedrooms ,looks like having highest price and lower the number of bedrooms has the lower number of prices.


# Blog Post Link:
https://medium.com/@honrao.pranav/interesting-facts-about-seattle-airbnb-usage-in-2016-7d3e20e81e82

# Reference:
1. https://www.kaggle.com/airbnb/seattle/home
2. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
3. http://python-graph-gallery.com/boxplot/#prettyPhoto
