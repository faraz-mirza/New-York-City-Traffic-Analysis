#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import requests

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from math import sin, cos, sqrt, atan2, radians

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, StringType, TimestampType, DateType, Row

import pyspark.sql.functions as func


# In[2]:


# Initializing the Spark Session
spark = SparkSession.builder.appName("Yellow Taxi").getOrCreate()


# In[3]:

# Schema of our Dataset
schema = StructType([
    StructField("vendor_id", StringType()),
    StructField("pickup_datetime", TimestampType()),
    StructField("dropoff_datetime", TimestampType()),
    StructField("passenger_count", IntegerType()),
    StructField("trip_distance", FloatType()),
    StructField("pickup_longitude", FloatType()),
    StructField("pickup_latitude", FloatType()),
    StructField("rate_code", StringType()),
    StructField("store_and_fwd_flag", StringType()),
    StructField("dropoff_longitude", FloatType()),
    StructField("dropoff_latitude", FloatType()),
    StructField("payment_type", StringType()),
    StructField("fare_amount", FloatType()),
    StructField("surcharge", FloatType()),
    StructField("mta_tax", FloatType()),
    StructField("tip_amount", FloatType()),
    StructField("tolls_amount", FloatType()),
    StructField("total_amount", FloatType())
])


# In[4]:


my_path = 'C:/Users/mirza/OneDrive/Documents/Spring 2019/MSA8050 - Scalable Data Analytics'
cluster_path = '/data/MSA_8050_Spring_19/7pm_2'
# Loading the data into our DataFrame
data = spark.read.csv(cluster_path + "/yellow_tripdata_2014-*.csv", header=True, schema=schema)


# In[ ]:


data.describe().toPandas().to_csv('/home/fmirza4/Quality Report.csv', index=False)


# In[5]:


data = data.fillna('N', subset=['store_and_fwd_flag'])


# In[6]:


# from pyspark.sql import DataFrameStatFunctions as statFunc

# quantiles_pickup_lat = statFunc(data).approxQuantile("pickup_latitude", [0.25,0.5,0.75], 0.1)
# quantiles_pickup_long = statFunc(data).approxQuantile("pickup_longitude", [0.25,0.5,0.75], 0.1)
# quantiles_dropoff_lat = statFunc(data).approxQuantile("dropoff_latitude", [0.25,0.5,0.75], 0.1)
# quantiles_dropoff_long = statFunc(data).approxQuantile("dropoff_longitude", [0.25,0.5,0.75], 0.1)
# print(quantiles_pickup_lat)
# print(quantiles_pickup_long)
# print(quantiles_dropoff_lat)
# print(quantiles_dropoff_long)


# In[6]:


# Removing Outliers in Long, Lat

data = data.filter((data.pickup_latitude>38.0) & (data.pickup_latitude < 43.0))
data = data.filter((data.pickup_longitude>-75.0) & (data.pickup_longitude < -72.0))
data = data.filter((data.dropoff_latitude>36.0) & (data.dropoff_latitude < 44.0))
data = data.filter((data.dropoff_longitude>-76.0) & (data.dropoff_longitude < -71.0))


# In[7]:


data.describe().toPandas().to_csv('/home/fmirza4/Quality Report_ver2.csv', index=False)


# In[7]:


full_sample_data = data


# In[8]:


#full_sample_data.dtypes


# In[ ]:


# Count of Rows
# full_sample_data.count()


# In[9]:


# Keeping the columns I need for my analysis
sample_data = full_sample_data.select(['pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_distance', 
                                       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                       'fare_amount', 'tip_amount', 'total_amount'])
sample_data.columns


# In[10]:


# Trimming the Long, Lat to 3 Decimal Places
sample_data = sample_data.withColumn("Pickup Trimmed Long", func.round(sample_data["pickup_longitude"], 3))
sample_data = sample_data.withColumn("Pickup Trimmed Lat", func.round(sample_data["pickup_latitude"], 3))
sample_data = sample_data.withColumn("Dropoff Trimmed Long", func.round("dropoff_longitude", 3))
sample_data = sample_data.withColumn("Dropoff Trimmed Lat", func.round("dropoff_latitude", 3))


# In[11]:


# Applying K-Means(12) on Pickup Trimmed Coordinates to get Zones/Neighbourhood 
vecAssembler = VectorAssembler(inputCols=['Pickup Trimmed Long', 'Pickup Trimmed Lat'], outputCol="features")
vector_df = vecAssembler.transform(sample_data)  # Vectorizing the features

k = 12
# for k in range(10,20):
#     kmeans = KMeans().setK(k).setSeed(1)
#     model = kmeans.fit(vector_df)
#     cost = model.computeCost(vector_df)
#     print(k, "Within Set Sum of Squared Errors = " + str(cost))

kmeans = KMeans().setK(k).setSeed(1)
model = kmeans.fit(vector_df)
transformed_data = model.transform(vector_df)


# In[12]:


# Extracting the Street Addresses and Zipcodes of Area Zones I got from K-Means
centers = model.clusterCenters()

street = list()
zipcode = list()
for center in centers:
    long, lat = round(center[0],3), round(center[1], 3)
    data = requests.get('https://nominatim.openstreetmap.org/reverse?format=json&lat={}&lon={}&zoom=18&addressdetails=1'.format(lat, long))
    json_data = json.loads(data.content)
    
    address = json_data['address']
    street.append(address.get('commercial', address.get('neighbourhood', address.get('suburb', address.get('road', None)))))
    postcode = address.get('postcode', None)
    if postcode:
        postcode = postcode.split(':')[0]
    print(postcode)
    zipcode.append(int(postcode))

sample_data = transformed_data.withColumn('Pickup Street Address', func.when(func.col('prediction')==0, street[0])                                .when(func.col('prediction')==1, street[1])                                .when(func.col('prediction')==2, street[2])                                .when(func.col('prediction')==3, street[3])                                .when(func.col('prediction')==4, street[4])                                .when(func.col('prediction')==5, street[5])                                .when(func.col('prediction')==6, street[6])                                .when(func.col('prediction')==7, street[7])                                .when(func.col('prediction')==8, street[8])                                .when(func.col('prediction')==9, street[9])                                .when(func.col('prediction')==10, street[10])                                .when(func.col('prediction')==11, street[11])                                .otherwise(None))

sample_data = sample_data.withColumn('Pickup Zip Code', func.when(func.col('prediction')==1, zipcode[0])                                .when(func.col('prediction')==2, zipcode[1])                                .when(func.col('prediction')==3, zipcode[2])                                .when(func.col('prediction')==4, zipcode[3])                                .when(func.col('prediction')==5, zipcode[4])                                .when(func.col('prediction')==6, zipcode[5])                                .when(func.col('prediction')==7, zipcode[6])                                .when(func.col('prediction')==8, zipcode[7])                                .when(func.col('prediction')==9, zipcode[8])                                .when(func.col('prediction')==10, zipcode[9])                                .when(func.col('prediction')==11, zipcode[10])                                .when(func.col('prediction')==12, zipcode[11])                                .otherwise(None))

sample_data = sample_data.drop('features')
sample_data = sample_data.withColumnRenamed("prediction", "pickup zone")


# In[13]:


import pandas as pd
my_table = list()
for i in range(len(centers)):
    my_table.append((centers[i][0], centers[i][1], street[i], zipcode[i]))


# In[14]:


# Repeating the K-means process with Dropoff Coordinates
vecAssembler = VectorAssembler(inputCols=['Dropoff Trimmed Long', 'Dropoff Trimmed Lat'], outputCol="features")
vector_df = vecAssembler.transform(sample_data)  # Vectorizing the features

k = 12
# for k in range(10,15):
#     kmeans = KMeans().setK(k).setSeed(1)
#     model = kmeans.fit(vector_df)
#     cost = model.computeCost(vector_df)
#     print(k, "Within Set Sum of Squared Errors = " + str(cost))

kmeans = KMeans().setK(k).setSeed(1)
model = kmeans.fit(vector_df)

transformed_data = model.transform(vector_df)
transformed_data = transformed_data.drop('features')


# In[15]:


centers = model.clusterCenters()

street = list()
zipcode = list()
for center in centers:
    long, lat = round(center[0],3), round(center[1], 3)
    data = requests.get('https://nominatim.openstreetmap.org/reverse?format=json&lat={}&lon={}&zoom=18&addressdetails=1'.format(lat, long))
    json_data = json.loads(data.content)
    address = json_data['address']
    street.append(address.get('commercial', address.get('neighbourhood', address.get('suburb', address.get('road', None)))))
    postcode = address.get('postcode', None)
    if postcode:
        postcode = postcode.split(':')[0]
    print(postcode)
    zipcode.append(int(postcode))

sample_data = transformed_data.withColumn('Dropoff Street Address', func.when(func.col('prediction')==0, street[0])                                .when(func.col('prediction')==1, street[1])                                .when(func.col('prediction')==2, street[2])                                .when(func.col('prediction')==3, street[3])                                .when(func.col('prediction')==4, street[4])                                .when(func.col('prediction')==5, street[5])                                .when(func.col('prediction')==6, street[6])                                .when(func.col('prediction')==7, street[7])                                .when(func.col('prediction')==8, street[8])                                .when(func.col('prediction')==9, street[9])                                .when(func.col('prediction')==10, street[10])                                .when(func.col('prediction')==11, street[11])                                .otherwise(None))

sample_data = sample_data.withColumn('Dropoff Zip Code', func.when(func.col('prediction')==0, zipcode[0])                                .when(func.col('prediction')==1, zipcode[1])                                .when(func.col('prediction')==2, zipcode[2])                                .when(func.col('prediction')==3, zipcode[3])                                .when(func.col('prediction')==4, zipcode[4])                                .when(func.col('prediction')==5, zipcode[5])                                .when(func.col('prediction')==6, zipcode[6])                                .when(func.col('prediction')==7, zipcode[7])                                .when(func.col('prediction')==8, zipcode[8])                                .when(func.col('prediction')==9, zipcode[9])                                .when(func.col('prediction')==10, zipcode[10])                                .when(func.col('prediction')==11, zipcode[11])                                .otherwise(None))

sample_data = sample_data.withColumnRenamed("prediction", "dropoff zone")


# In[ ]:


# sample_data.select('pickup zone').distinct().show()


# In[16]:


for i in range(len(centers)):
    my_table.append((centers[i][0], centers[i][1], street[i], zipcode[i]))

cluster = pd.DataFrame(my_table, columns=['Longitude', 'Latitude', 'Area', 'Zipcode'])
cluster.drop_duplicates('Area', inplace=True)

cluster.to_csv('/home/fmirza4/Clusters.csv', index=False)
cluster.head(20)


# In[ ]:


# def get_distance(lat1, lon1, lat2, lon2):
#     R = 6373.0 # approximate radius of earth in km

#     lat1 = radians(lat1)
#     lon1 = radians(lon1)
#     lat2 = radians(lat2)
#     lon2 = radians(lon2)

#     dlon = lon2 - lon1
#     dlat = lat2 - lat1

#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))

#     distance = R * c * 0.62137

#     return float(distance)


# In[17]:


# # Computing the Manhattan Distance of Rides
# distances = sample_data.select('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')\
#                             .rdd.map(lambda x: Row(get_distance(x[0], x[1], x[2], x[3]))).toDF()

# distances = distances.select(func.col("_1").alias("Manhattan Distance"))

# distances = distances.withColumn("rowId", func.monotonically_increasing_id())
# sample_data = sample_data.withColumn("rowId", func.monotonically_increasing_id())

# sample_data = sample_data.join(distances, on=['rowId'])
# sample_data.take(1)


# In[10]:


# Extracting the Day of the week and the Actual Day from the Date

sample_data = sample_data.withColumn("DoW", func.date_format("pickup_datetime",'u').cast(IntegerType()))
sample_data = sample_data.withColumn("Day", func.date_format("pickup_datetime",'E'))


# In[11]:


# Calculating the Duration of Rides
timeFmt = "yyyy-MM-dd'T'HH:mm:ss"
secsPerRide = (func.unix_timestamp("dropoff_datetime", format=timeFmt) - func.unix_timestamp("pickup_datetime", format=timeFmt))
minsPerRide = secsPerRide / 60

sample_data = sample_data.withColumn("Duration", minsPerRide.cast(FloatType()))


# In[20]:


sample_data = sample_data.withColumn('Pickup Street Address', func.when(func.col('Pickup Zip Code')==11420, 'JFK Terminal').otherwise(func.col("Pickup Street Address")))
sample_data = sample_data.withColumn('Dropoff Street Address', func.when(func.col('Dropoff Zip Code')==11420, 'JFK Terminal').otherwise(func.col("Dropoff Street Address")))


# In[35]:


# Popular Locations 
popular_trips = sample_data.groupBy("Pickup Street Address", "Dropoff Street Address").count().orderBy('count', ascending=False)
popular_trips = popular_trips.toPandas()
popular_trips = popular_trips[popular_trips['Pickup Street Address'] != popular_trips['Dropoff Street Address']]

popular_trips.to_csv('/home/fmirza4/Popular Trips.csv', index=False)


# In[13]:


df = sample_data.withColumn('Cost', sample_data['total_amount'] - sample_data['tip_amount'])


# In[12]:


avg_counts_per_week = df.groupBy("DoW").agg(func.count('pickup_latitude'))
avg_counts_per_week.toPandas().to_csv('/home/fmirza4/avg_counts_per_week.csv', index=False)


# In[22]:


# Average Duration of a Trip based on the Location and Day of the Week
avg_duration_per_day = df.groupBy("Pickup Street Address", "Dropoff Street Address", "DoW", "Day").agg(func.avg('Duration').alias('Average duration'), func.avg('trip_distance'), func.avg('fare_amount'), func.avg('Cost')).orderBy("Pickup Street Address", "Dropoff Street Address", "DoW")
avg_duration_per_day = avg_duration_per_day.toPandas()
avg_duration_per_day = avg_duration_per_day[avg_duration_per_day['Pickup Street Address'] != avg_duration_per_day['Dropoff Street Address']]
avg_duration_per_day.to_csv('/home/fmirza4/Location_Daily_Analysis_Trips_final.csv', index=False)


# In[23]:


# Average Duration of a Trip based on the Location and Hour of the Day
avg_duration_per_hour = df.groupBy('Pickup Street Address', 'Dropoff Street Address', func.hour('pickup_datetime').alias('hour')).agg(func.avg('Duration'), func.avg('trip_distance'), func.avg('fare_amount'), func.avg('Cost')).orderBy("Pickup Street Address", "Dropoff Street Address", "hour")
avg_duration_per_hour = avg_duration_per_hour.toPandas()
avg_duration_per_hour = avg_duration_per_hour[avg_duration_per_hour['Pickup Street Address'] != avg_duration_per_hour['Dropoff Street Address']]
avg_duration_per_hour.to_csv('/home/fmirza4/Location_Hourly_Analysis_Trips_final.csv', index=False)


# In[26]:


# Count of Rides per Day
agg_day = df.select(func.date_format('pickup_datetime', 'MM/dd/yyyy').alias('Date')).groupBy('Date').count().orderBy('Date')
agg_day.toPandas().to_csv('/home/fmirza4/RidesPerDay.csv', index=False)


# In[27]:


# Count of Rides per Hour
hour_count = df.groupBy(func.hour("pickup_datetime").alias("hour")).count().orderBy('count', ascending=False)
hour_count.toPandas().to_csv('/home/fmirza4/RidesPerHour.csv', index=False)


# In[28]:


# General Average Duration of a Ride per hour of the Day
avg_duration_ride_per_hour = df.groupBy(func.hour("pickup_datetime").alias('Hour')).agg(func.avg('duration').alias('Average Ride Duration')).orderBy('Average Ride Duration', ascending=False)
avg_duration_ride_per_hour.toPandas().to_csv('/home/fmirza4/Average Ride Duration per Hour.csv', index=False)


# In[29]:


# General Average Duration of a Ride per Day of the Week
avg_duration_ride_per_day = df.groupBy("DoW", "Day").agg(func.avg('duration').alias('Average Duration'), ).orderBy('Average Duration', ascending=False)
avg_duration_ride_per_day.toPandas().to_csv('/home/fmirza4/Average Ride Duration per Day.csv', index=False)




