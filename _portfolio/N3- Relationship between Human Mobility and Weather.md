###Project Summary
<img src ='https://images.unsplash.com/photo-1548859047-1d15def63a14?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=2700&q=80' width="800" height = "200">
######Dr. Wei Zhai (Assistant Professor at Hong Kong Baptist University) and Ivy(Ruxin) Tong | Oct 15th, 2021 | Image courtesy of Ivan Olenkevich

####**Objective** : Evaluation and Prediction of American’s Mobility under Extreme Weather Events using Artificial Intelligence (Microsoft AI for Earth Grant)


####**Datasets** :  Weather_CBG_2019 | Social Distancing Metrics v2.1 | Top 100 CBG City
######***About the data***
- ***Dataset - Weather_CBG_2019 records daily weather parameters by census block group in the United States. There're 7 variables*** 

| Variable   | Description  | Unit | 
|-----------------  |---------------|------|
| geoid    | census block group ID |  | 
| precip  | daily precipitation | mm|
| rmax | maximum daily relative humidity | % |
| rmin | mminimum daily relative humidity | % |
| srad | surface downwelling solar radiation | W/m^2 |
| tmax | maximum daily temperature | degress F |
| tmin | minimum daily temperature | degress F |
| wind_speed | wind speed | mph|


- ***Dataset - Social Distancing Metrics v2.1 is a product of Safegraph which aggregately summarizes daily views of USA foot-traffic between census block groups. There're 23 variables. For this analysis, I mainly use ***

     *description based on Safegraph documentation*
  
| Variable   | Description  | Unit | 
|-----------------  |---------------|------|
| origin_census_block_group    | 12-digit FIPS code for the Census Block Group |  | 
| distance_traveled_from_home  | Median distance traveled from the geohash-7 of the home by the devices measured within the time period. All distance with 0 has been excluded. | meter |
| date_range_start  | Start time for measurement | YYYY-MM-DDTHH:mm:SS±hh:mm |
| date_range_end  | End time for measurement  | YYYY-MM-DDTHH:mm:SS±hh:mm|
| device_count | Total number of devices seen during the date range | count  |
| median_home_dwell_time  | Median dwell time at home during the time period |  min |  
| completely_home_device_count  | Number of device devices do not leave the house during the time period | count  |  

- ***Dataset - Top 100 CBG City. There're 4 variables*** 

| Variable   | Description  |
|-----------------  |---------------|
| geoid10    | census block group ID | 
| name  | name of city |
| class | municipality |
| st | state abbreviation |






- ***Additional variables created for the analysis ***


| Variable   | Description  | Unit | 
|-----------------  |---------------|------|
| month   | month of each year | 1-12 | 
| day  | day of each year | 1-31 |
| weekday  | day of each week | 0-6 (0 for Mon) |
| ratio of not leaving | completely_home_device_count / device_count | ratio between 0 and 1 |

  
  
######***Acknowledgements*** : Social Distancing Metrics v2.1 is downloaded "from [SafeGraph](https://www.safegraph.com/), a data company that aggregates anonymized location data from numerous applications in order to provide insights about physical places, via the [Placekey](https://www.placekey.io/) Community. To enhance privacy, SafeGraph excludes census block group information if fewer than five devices visited an establishment in a month from a given census block group.”




####**Conclusion**:




####**Model and Result**:


####**Notebook** : There are two notebooks for this project:

Notebook1 details the data analysis for 2019 weather dataset, Jan social distancing dataset, and initial Jan modeling (mainly in Python) [Current]   

Notebook2 idetails 2019 whole year modeling (mainly in Spark)

#### Analysis
###### Step 0 : Install and Import libraries


```python
%run ./Packages_setup
```

###### Step 1 : Data Preparation and Analysis for weather_cbg_2019
  - Secure access to  AWS s3 bucket and read the dataset


```python
# # Mount a bucket using AWS keys (write and read access to all the objects in the S3 buckets). Alternatively, use AWS instance profile)
awsAccessKey = ""
secretKey = "".replace("/", "%2F")
awsBucketName = "weathercbg2019"
mountPoint = f"/mnt/weathercbg"

mountTarget = "s3a://{}:{}@{}".format(awsAccessKey, secretKey, awsBucketName)
dbutils.fs.mount(mountTarget, mountPoint)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
dbutils.fs.ls("/mnt/weathercbg")
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
dbutils.fs.head("dbfs:/mnt/weathercbg/weather_cbg_2019.csv",1000)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
# # #read weathercbgDF
weathercbgDF = (spark.read
  .option("delimiter", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv("/mnt/weathercbg/weather_cbg_2019.csv")
 )
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>


- Transform and Explore the weather dataset


```python
# #  Add three date-related variables
weathercbgDF = weathercbgDF.withColumn("month", F.month('date'))
weathercbgDF = weathercbgDF.withColumn("day", F.dayofmonth('date'))
weathercbgDF = weathercbgDF.withColumn("year", F.year('date'))
weathercbgDF.show(5)
# save to local folder so that we can access directly through databricks
weathercbgDF.write.format('com.databricks.spark.csv').save("/FileStore/weathercbgDF.csv",header = 'true',inferSchema = "true")
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
#   unmount s3
try:
  dbutils.fs.unmount(mountPoint)
except:
  print("{} already unmounted".format(mountPoint))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
#decide to add a new variable after unmounting to s3
weathercbgDF = spark.read.format("csv").option("header", "true").option("inferSchema","true").load("/FileStore/weathercbgDF.csv")
# weathercbgDF =weathercbgDF.withColumn('day_of_week',dayofweek(weathercbgDF.date))

def spark_df_shape(self):
    return (self.count(),len(self.columns)) 
pyspark.sql.dataframe.DataFrame.shape = spark_df_shape


print("weather cbg 2019 dataset")
weathercbgDF.printSchema()
print("shape:")
print(weathercbgDF.shape())
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">weather cbg 2019 dataset
root
-- geoid: long (nullable = true)
-- date: string (nullable = true)
-- precip: double (nullable = true)
-- rmax: double (nullable = true)
-- rmin: double (nullable = true)
-- srad: double (nullable = true)
-- tmin: double (nullable = true)
-- tmax: double (nullable = true)
-- wind_speed: double (nullable = true)
-- month: integer (nullable = true)
-- day: integer (nullable = true)
-- year: integer (nullable = true)

shape:
(78654945, 12)
</div>



```python
from pyspark.sql.functions import min, max

display(
  weathercbgDF.select(min("date"), max("date"))
)
```


<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>min(date)</th><th>max(date)</th></tr></thead><tbody><tr><td>2019-01-01</td><td>2019-12-31</td></tr></tbody></table></div>


- Create summary table group by month and geoid & Basic visualization


```python
#   create monthly summary table group by geoid, for visualization and faster operation
weather_cbg_month = (weathercbgDF.groupBy("geoid","month")
                            .agg(
                              mean("precip").alias("mean_precip"),
                              mean("rmax").alias("mean_rmax"),
                              mean("rmin").alias("mean_rmin"),
                              mean("srad").alias("mean_srad"),
                              mean("tmin").alias("mean_tmin"),
                              mean("tmax").alias("mean_tmax"),
                              mean("wind_speed").alias("mean_wind_speed"))
                             .sort(["geoid","month"]))

weather_cbg_month.show(5)

weather_cbg_month.write.format('com.databricks.spark.csv').save("/FileStore/weather_cbg_month.csv",header = 'true',inferSchema = 'true')
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
## Generate a sample for exploratory data analysis
weather_cbg_month = spark.read.format("csv").option("header", "true",).option("inferSchema", "true").load("/FileStore/weather_cbg_month.csv") 
weather_cbg_month.printSchema()

weather_cbg_2019_month_sample =weather_cbg_month.sample(0.01).toPandas()
weather_cbg_2019_month_sample.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geoid</th>
      <th>month</th>
      <th>mean_precip</th>
      <th>mean_rmax</th>
      <th>mean_rmin</th>
      <th>mean_srad</th>
      <th>mean_tmin</th>
      <th>mean_tmax</th>
      <th>mean_wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>360550117054</td>
      <td>2</td>
      <td>2.086</td>
      <td>84.064</td>
      <td>48.271</td>
      <td>111.157</td>
      <td>20.011</td>
      <td>36.018</td>
      <td>11.976</td>
    </tr>
    <tr>
      <th>1</th>
      <td>360550121004</td>
      <td>1</td>
      <td>2.106</td>
      <td>90.106</td>
      <td>57.297</td>
      <td>80.471</td>
      <td>15.710</td>
      <td>29.553</td>
      <td>12.289</td>
    </tr>
    <tr>
      <th>2</th>
      <td>360550123012</td>
      <td>2</td>
      <td>2.150</td>
      <td>84.343</td>
      <td>49.146</td>
      <td>110.461</td>
      <td>20.191</td>
      <td>36.018</td>
      <td>12.279</td>
    </tr>
    <tr>
      <th>3</th>
      <td>360550123012</td>
      <td>3</td>
      <td>1.287</td>
      <td>85.400</td>
      <td>45.123</td>
      <td>162.816</td>
      <td>23.386</td>
      <td>41.903</td>
      <td>11.055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>360550123041</td>
      <td>4</td>
      <td>2.353</td>
      <td>85.810</td>
      <td>46.040</td>
      <td>216.390</td>
      <td>36.272</td>
      <td>56.144</td>
      <td>10.484</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Understand distribution of Variables
variables = weather_cbg_2019_month_sample.drop(["geoid","month"], axis = 1)
fig,ax = plt.subplots(1,7, figsize= (30,5))

for i, variable in enumerate(variables):
  sns.distplot(variables[variable], ax=ax[i])
```


    
![png](output_17_0.png)
    



```python
# these distributions are not on the same scale, standardize them
fix, ax = plt.subplots(1,7,figsize = (25,5))

for i, variable in enumerate(variables):
  original_variables = variables[variable]
  variable_scaled = (original_variables - original_variables.mean())/original_variables.std()
  sns.distplot(variable_scaled, ax=ax[i])
  ax[i].set_xlim(-2,2)
```


    
![png](output_18_0.png)
    



```python
#correlation
g = sns.pairplot(variables)
g.fig.set_size_inches(15,10)
```


    
![png](output_19_0.png)
    



```python
corr = variables.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask,0)] = True
sns.heatmap(corr, mask = mask,square = True, annot = True)
```


    
![png](output_20_0.png)
    



```python
# # scaled data by month. This is generally the US case, it varies by the region
for i, variable in enumerate(variables):
  original_variables = variables[variable]
  variable_scaled = (original_variables - original_variables.mean())/original_variables.std()
  sns.lineplot(x = "month", y=variable_scaled, data = weather_cbg_2019_month_sample,legend='brief', label=variable)

plt.ylabel("values")
plt.legend(bbox_to_anchor=(1.02, 1),borderaxespad=0.)
plt.title("mean weather parameters level by month")

```


    
![png](output_21_0.png)
    


- Prepare Jan file for initial modeling


```python
# Prepare Jan file for initial modeling
weatherDF_Jan = weathercbgDF.where(col("month") == 1).toPandas()
weatherDF_Jan.head(10)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[5]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geoid</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>month</th>
      <th>day</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10730059033</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>97.200</td>
      <td>65.200</td>
      <td>111.800</td>
      <td>48.470</td>
      <td>61.970</td>
      <td>2.461</td>
      <td>1</td>
      <td>1</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10730059033</td>
      <td>2019-01-02</td>
      <td>20.900</td>
      <td>86.300</td>
      <td>81.600</td>
      <td>30.600</td>
      <td>48.470</td>
      <td>52.790</td>
      <td>3.355</td>
      <td>1</td>
      <td>2</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10730059033</td>
      <td>2019-01-03</td>
      <td>24.900</td>
      <td>79.400</td>
      <td>63.100</td>
      <td>59.200</td>
      <td>50.630</td>
      <td>59.450</td>
      <td>5.816</td>
      <td>1</td>
      <td>3</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10730059033</td>
      <td>2019-01-04</td>
      <td>0.900</td>
      <td>100.000</td>
      <td>55.700</td>
      <td>99.100</td>
      <td>40.550</td>
      <td>62.150</td>
      <td>14.316</td>
      <td>1</td>
      <td>4</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10730059033</td>
      <td>2019-01-05</td>
      <td>0.000</td>
      <td>87.600</td>
      <td>39.200</td>
      <td>126.800</td>
      <td>36.950</td>
      <td>60.170</td>
      <td>7.382</td>
      <td>1</td>
      <td>5</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10730059033</td>
      <td>2019-01-06</td>
      <td>0.000</td>
      <td>95.800</td>
      <td>33.200</td>
      <td>119.400</td>
      <td>36.770</td>
      <td>66.830</td>
      <td>2.908</td>
      <td>1</td>
      <td>6</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10730059033</td>
      <td>2019-01-07</td>
      <td>0.400</td>
      <td>83.800</td>
      <td>38.300</td>
      <td>126.500</td>
      <td>44.150</td>
      <td>67.370</td>
      <td>7.158</td>
      <td>1</td>
      <td>7</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10730059033</td>
      <td>2019-01-08</td>
      <td>0.000</td>
      <td>100.000</td>
      <td>43.900</td>
      <td>116.400</td>
      <td>42.710</td>
      <td>68.990</td>
      <td>8.948</td>
      <td>1</td>
      <td>8</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10730059033</td>
      <td>2019-01-09</td>
      <td>0.000</td>
      <td>68.600</td>
      <td>34.000</td>
      <td>135.000</td>
      <td>31.730</td>
      <td>51.530</td>
      <td>10.961</td>
      <td>1</td>
      <td>9</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10730059033</td>
      <td>2019-01-10</td>
      <td>0.000</td>
      <td>60.100</td>
      <td>31.100</td>
      <td>133.500</td>
      <td>27.410</td>
      <td>45.590</td>
      <td>8.053</td>
      <td>1</td>
      <td>10</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(weatherDF_Jan.shape)
print(weatherDF_Jan.geoid.nunique())
weatherDF_Jan.info()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[14]: (6680283, 12)</div>



```python
weatherDF_Jan[weatherDF_Jan.geoid == 40130101022].head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geoid</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>month</th>
      <th>day</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5906078</th>
      <td>40130101022</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>98.100</td>
      <td>45.900</td>
      <td>131.500</td>
      <td>24.530</td>
      <td>43.430</td>
      <td>4.474</td>
      <td>1</td>
      <td>1</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5906079</th>
      <td>40130101022</td>
      <td>2019-01-02</td>
      <td>0.000</td>
      <td>69.500</td>
      <td>25.000</td>
      <td>135.900</td>
      <td>23.810</td>
      <td>46.850</td>
      <td>3.803</td>
      <td>1</td>
      <td>2</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5906080</th>
      <td>40130101022</td>
      <td>2019-01-03</td>
      <td>0.000</td>
      <td>53.200</td>
      <td>16.900</td>
      <td>140.200</td>
      <td>30.650</td>
      <td>54.950</td>
      <td>3.132</td>
      <td>1</td>
      <td>3</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5906081</th>
      <td>40130101022</td>
      <td>2019-01-04</td>
      <td>0.000</td>
      <td>46.000</td>
      <td>12.700</td>
      <td>140.900</td>
      <td>37.310</td>
      <td>63.770</td>
      <td>3.803</td>
      <td>1</td>
      <td>4</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>5906082</th>
      <td>40130101022</td>
      <td>2019-01-05</td>
      <td>18.900</td>
      <td>58.700</td>
      <td>19.700</td>
      <td>98.500</td>
      <td>39.470</td>
      <td>64.310</td>
      <td>7.158</td>
      <td>1</td>
      <td>5</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>


###### Step 2 : Data Preparation and Analysis for Social Distancing Metrics v2.1

- Get data from Safegraph
    - Download this dataset from safegraph through cluster terminal to filepath 'cd databricks/driver/mnt/' -> Move them to '/dbfs/mnt#'


```python
# use this command to move files so that I can directly access from databricks data UI
dbutils.fs.mv(r"file:/mnt/", r"dbfs:/social_distance_data/", True)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
%fs ls /social_distance_data/social_distance/2019/
```


<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>path</th><th>name</th><th>size</th></tr></thead><tbody><tr><td>dbfs:/social_distance_data/social_distance/2019/01/</td><td>01/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/02/</td><td>02/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/03/</td><td>03/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/04/</td><td>04/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/05/</td><td>05/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/06/</td><td>06/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/07/</td><td>07/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/08/</td><td>08/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/09/</td><td>09/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/10/</td><td>10/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/11/</td><td>11/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/12/</td><td>12/</td><td>0</td></tr></tbody></table></div>



```python
%fs ls /social_distance_data/social_distance/2019/01
```


<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>path</th><th>name</th><th>size</th></tr></thead><tbody><tr><td>dbfs:/social_distance_data/social_distance/2019/01/01/</td><td>01/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/02/</td><td>02/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/03/</td><td>03/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/04/</td><td>04/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/05/</td><td>05/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/06/</td><td>06/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/07/</td><td>07/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/08/</td><td>08/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/09/</td><td>09/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/10/</td><td>10/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/11/</td><td>11/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/12/</td><td>12/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/13/</td><td>13/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/14/</td><td>14/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/15/</td><td>15/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/16/</td><td>16/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/17/</td><td>17/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/18/</td><td>18/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/19/</td><td>19/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/20/</td><td>20/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/21/</td><td>21/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/22/</td><td>22/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/23/</td><td>23/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/24/</td><td>24/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/25/</td><td>25/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/26/</td><td>26/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/27/</td><td>27/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/28/</td><td>28/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/29/</td><td>29/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/30/</td><td>30/</td><td>0</td></tr><tr><td>dbfs:/social_distance_data/social_distance/2019/01/31/</td><td>31/</td><td>0</td></tr></tbody></table></div>



```python
%fs ls /social_distance_data/social_distance/2019/01/01
```


<style scoped>
  .table-result-container {
    max-height: 300px;
    overflow: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
  }
  th {
    text-align: left;
  }
</style><div class='table-result-container'><table class='table-result'><thead style='background-color: white'><tr><th>path</th><th>name</th><th>size</th></tr></thead><tbody><tr><td>dbfs:/social_distance_data/social_distance/2019/01/01/2019-01-01-social-distancing.csv.gz</td><td>2019-01-01-social-distancing.csv.gz</td><td>90505721</td></tr></tbody></table></div>


- Preparing SafeGraph social-distancing data
    - read one gzip file first to understand panda dataframe
    - read January data and prepare it for initial modeling, reason being that i wanna test the model out before scaling up the whole dataset
    - read all 2019 social-distancing dataset (approx 800M rows)


```python
# use panda to read one file 
sd20190101 = pd.read_csv("/dbfs/social_distance_data/social_distance/2019/01/01/2019-01-01-social-distancing.csv.gz",compression = "gzip")
sd20190101.head() 
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>date_range_start</th>
      <th>date_range_end</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>bucketed_distance_traveled</th>
      <th>median_dwell_at_bucketed_distance_traveled</th>
      <th>completely_home_device_count</th>
      <th>median_home_dwell_time</th>
      <th>bucketed_home_dwell_time</th>
      <th>at_home_by_each_hour</th>
      <th>part_time_work_behavior_devices</th>
      <th>full_time_work_behavior_devices</th>
      <th>destination_cbgs</th>
      <th>delivery_behavior_devices</th>
      <th>median_non_home_dwell_time</th>
      <th>candidate_device_count</th>
      <th>bucketed_away_from_home_time</th>
      <th>median_percentage_time_home</th>
      <th>bucketed_percentage_time_home</th>
      <th>mean_home_dwell_time</th>
      <th>mean_non_home_dwell_time</th>
      <th>mean_distance_traveled_from_home</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>78</td>
      <td>687</td>
      <td>{"16001-50000":14,"0":28,"&gt;50000":8,"2001-8000...</td>
      <td>{"16001-50000":38,"&gt;50000":207,"&lt;1000":52,"200...</td>
      <td>28</td>
      <td>714</td>
      <td>{"721-1080":14,"361-720":14,"61-360":5,"&lt;60":1...</td>
      <td>[42,44,45,46,46,45,42,44,44,41,48,39,37,33,30,...</td>
      <td>7</td>
      <td>1</td>
      <td>{"281419502003":1,"010330210003":5,"2401503010...</td>
      <td>1</td>
      <td>52</td>
      <td>179</td>
      <td>{"21-45":3,"481-540":1,"541-600":1,"46-60":4,"...</td>
      <td>92</td>
      <td>{"0-25":17,"76-100":51,"51-75":5,"26-50":1}</td>
      <td>721</td>
      <td>209</td>
      <td>134263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10730049022</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>259</td>
      <td>5381</td>
      <td>{"16001-50000":12,"0":94,"&gt;50000":58,"2001-800...</td>
      <td>{"16001-50000":165,"&gt;50000":177,"&lt;1000":129,"2...</td>
      <td>93</td>
      <td>58</td>
      <td>{"721-1080":22,"361-720":23,"61-360":57,"&lt;60":...</td>
      <td>[48,53,56,58,57,62,59,63,63,62,61,62,56,57,54,...</td>
      <td>13</td>
      <td>2</td>
      <td>{"121030273154":1,"132150004003":1,"0107301310...</td>
      <td>1</td>
      <td>44</td>
      <td>1312</td>
      <td>{"21-45":16,"481-540":5,"541-600":7,"46-60":8,...</td>
      <td>53</td>
      <td>{"0-25":112,"76-100":117,"51-75":12,"26-50":14}</td>
      <td>292</td>
      <td>221</td>
      <td>76229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11210118001</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>91</td>
      <td>1591</td>
      <td>{"16001-50000":5,"0":46,"&gt;50000":10,"2001-8000...</td>
      <td>{"16001-50000":84,"&gt;50000":247,"&lt;1000":181,"20...</td>
      <td>45</td>
      <td>487</td>
      <td>{"721-1080":13,"361-720":17,"61-360":13,"&lt;60":...</td>
      <td>[40,40,42,41,36,41,40,41,41,37,39,39,34,36,34,...</td>
      <td>3</td>
      <td>1</td>
      <td>{"011150405022":2,"011210118002":7,"0103796110...</td>
      <td>1</td>
      <td>0</td>
      <td>299</td>
      <td>{"21-45":3,"541-600":1,"46-60":1,"721-840":1,"...</td>
      <td>100</td>
      <td>{"0-25":25,"76-100":53,"51-75":6,"26-50":4}</td>
      <td>588</td>
      <td>181</td>
      <td>15741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11250106021</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>392</td>
      <td>4623</td>
      <td>{"16001-50000":55,"0":172,"&gt;50000":32,"2001-80...</td>
      <td>{"16001-50000":61,"&gt;50000":86,"&lt;1000":62,"2001...</td>
      <td>171</td>
      <td>860</td>
      <td>{"721-1080":60,"361-720":50,"61-360":46,"&lt;60":...</td>
      <td>[212,225,231,240,238,234,239,235,229,229,227,2...</td>
      <td>26</td>
      <td>8</td>
      <td>{"010730142031":4,"010730141021":2,"1313505054...</td>
      <td>1</td>
      <td>20</td>
      <td>1059</td>
      <td>{"21-45":26,"481-540":2,"541-600":7,"46-60":14...</td>
      <td>97</td>
      <td>{"0-25":66,"76-100":274,"51-75":24,"26-50":22}</td>
      <td>770</td>
      <td>172</td>
      <td>12937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21880002003</td>
      <td>2019-01-01T00:00:00-09:00</td>
      <td>2019-01-02T00:00:00-09:00</td>
      <td>10</td>
      <td>0</td>
      <td>{"1-1000":1,"&gt;50000":2,"0":1}</td>
      <td>{"&gt;50000":33,"&lt;1000":10}</td>
      <td>3</td>
      <td>1197</td>
      <td>{"&gt;1080":5,"&lt;60":1,"61-360":1}</td>
      <td>[1,3,4,3,7,4,5,5,6,4,6,6,4,7,7,5,5,3,8,1,6,8,5,5]</td>
      <td>1</td>
      <td>1</td>
      <td>{"021880002003":6,"550790217005":1,"0218800020...</td>
      <td>1</td>
      <td>10</td>
      <td>36</td>
      <td>{"&lt;20":5,"601-660":1,"361-420":1,"841-960":1}</td>
      <td>99</td>
      <td>{"0-25":4,"76-100":5,"26-50":1}</td>
      <td>857</td>
      <td>241</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(sd20190101.shape)
sd20190101.info()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[22]: (219490, 23)</div>



```python
#lets' take a look at the destination column
!pip install -q --upgrade git+https://github.com/SafeGraphInc/safegraph_py
from safegraph_py_functions.safegraph_py_functions import *
destination = sd20190101[['origin_census_block_group','destination_cbgs']]
unpack_json_and_merge_fast(destination, json_column='destination_cbgs', key_col_name='destination_cbgs_exploded', value_col_name='destination_cbgs_count', chunk_n= 1000).drop('destination_cbgs',axis = 1).head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>destination_cbgs_exploded</th>
      <th>destination_cbgs_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>281419502003</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10330210004</td>
      <td>010330210003</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10330210004</td>
      <td>240150301002</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10330210004</td>
      <td>471150502012</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10330210004</td>
      <td>010770102002</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
#bucket distance travel
bucket_distance = sd20190101[['origin_census_block_group','bucketed_distance_traveled']]
unpack_json_and_merge_fast(bucket_distance, json_column='bucketed_distance_traveled', key_col_name='bucketed_distance_traveled_exploded', value_col_name='bucketed_distance_traveled_count', chunk_n= 1000).drop('bucketed_distance_traveled',axis = 1).head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>bucketed_distance_traveled_exploded</th>
      <th>bucketed_distance_traveled_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>16001-50000</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10330210004</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10330210004</td>
      <td>&gt;50000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10330210004</td>
      <td>2001-8000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10330210004</td>
      <td>1-1000</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>


- Initial Modeling for Jan data only - Part 1
   - Prepare Jan social distancing dataset


```python
# Prepare January file for initial modeling
Jan_files = glob.glob("/dbfs/social_distance_data/social_distance/2019/01/*/*")

li= []
for file in Jan_files:
    df = pd.read_csv(file, compression='gzip', usecols=['origin_census_block_group','date_range_start','date_range_end','distance_traveled_from_home','mean_home_dwell_time','completely_home_device_count','device_count'])
    li.append(df)

sd_data_jan = pd.concat(li, axis=0, ignore_index=True)
del li
sd_data_jan.head(10)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>date_range_start</th>
      <th>date_range_end</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>78</td>
      <td>687</td>
      <td>28</td>
      <td>721</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10730049022</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>259</td>
      <td>5381</td>
      <td>93</td>
      <td>292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11210118001</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>91</td>
      <td>1591</td>
      <td>45</td>
      <td>588</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11250106021</td>
      <td>2019-01-01T00:00:00-06:00</td>
      <td>2019-01-02T00:00:00-06:00</td>
      <td>392</td>
      <td>4623</td>
      <td>171</td>
      <td>770</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21880002003</td>
      <td>2019-01-01T00:00:00-09:00</td>
      <td>2019-01-02T00:00:00-09:00</td>
      <td>10</td>
      <td>0</td>
      <td>3</td>
      <td>857</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40019450014</td>
      <td>2019-01-01T00:00:00-07:00</td>
      <td>2019-01-02T00:00:00-07:00</td>
      <td>97</td>
      <td>615</td>
      <td>33</td>
      <td>204</td>
    </tr>
    <tr>
      <th>6</th>
      <td>40130101023</td>
      <td>2019-01-01T00:00:00-07:00</td>
      <td>2019-01-02T00:00:00-07:00</td>
      <td>135</td>
      <td>3635</td>
      <td>47</td>
      <td>687</td>
    </tr>
    <tr>
      <th>7</th>
      <td>40132175012</td>
      <td>2019-01-01T00:00:00-07:00</td>
      <td>2019-01-02T00:00:00-07:00</td>
      <td>79</td>
      <td>1205</td>
      <td>55</td>
      <td>799</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40136145004</td>
      <td>2019-01-01T00:00:00-07:00</td>
      <td>2019-01-02T00:00:00-07:00</td>
      <td>89</td>
      <td>1435</td>
      <td>28</td>
      <td>802</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40136170001</td>
      <td>2019-01-01T00:00:00-07:00</td>
      <td>2019-01-02T00:00:00-07:00</td>
      <td>59</td>
      <td>2557</td>
      <td>23</td>
      <td>821</td>
    </tr>
  </tbody>
</table>
</div>



```python
sd_data_jan['date_range_start'] = pd.to_datetime(sd_data_jan['date_range_start'],utc= True)
sd_data_jan['date_range_end'] = pd.to_datetime(sd_data_jan['date_range_end'],utc= True)
sd_data_jan['month'] = sd_data_jan['date_range_start'].dt.month
sd_data_jan['day'] = sd_data_jan['date_range_start'].dt.day
sd_data_jan['dayofweek'] = sd_data_jan['date_range_start'].dt.dayofweek
sd_data_jan['date_time'] = sd_data_jan['date_range_end']-sd_data_jan['date_range_start']
# sd_data_jan['date'] = [d.date() for d in sd_data_jan['date_range_start']]
sd_data_jan['ratio_not_leaving'] = round(sd_data_jan['completely_home_device_count']/sd_data_jan['device_count'],4)
sd_data_jan = sd_data_jan.drop(['date_range_start','date_range_end'],1)

# sd_data_jan.to_csv("/dbfs/social_distance_data/jan.csv",index= False)  # save it for next time use
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
sd_data_jan = pd.read_csv("/dbfs/social_distance_data/jan.csv")
sd_data_jan.head()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[6]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>date_time</th>
      <th>ratio_not_leaving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>78</td>
      <td>687</td>
      <td>28</td>
      <td>721</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1 days 00:00:00.000000000</td>
      <td>0.359</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10730049022</td>
      <td>259</td>
      <td>5381</td>
      <td>93</td>
      <td>292</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1 days 00:00:00.000000000</td>
      <td>0.359</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11210118001</td>
      <td>91</td>
      <td>1591</td>
      <td>45</td>
      <td>588</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1 days 00:00:00.000000000</td>
      <td>0.494</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11250106021</td>
      <td>392</td>
      <td>4623</td>
      <td>171</td>
      <td>770</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1 days 00:00:00.000000000</td>
      <td>0.436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21880002003</td>
      <td>10</td>
      <td>0</td>
      <td>3</td>
      <td>857</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1 days 00:00:00.000000000</td>
      <td>0.300</td>
    </tr>
  </tbody>
</table>
</div>



```python
sd_data_jan.shape
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[47]: (6805915, 10)</div>



```python
sd_data_jan[sd_data_jan.date_time > '1 days 00:00:00']
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>date_time</th>
      <th>ratio_not_leaving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4297985</th>
      <td>21989401001</td>
      <td>20</td>
      <td>0</td>
      <td>15</td>
      <td>954</td>
      <td>1</td>
      <td>20</td>
      <td>6</td>
      <td>1 days 01:00:00</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>4315583</th>
      <td>21989401002</td>
      <td>61</td>
      <td>155</td>
      <td>33</td>
      <td>817</td>
      <td>1</td>
      <td>20</td>
      <td>6</td>
      <td>1 days 01:00:00</td>
      <td>0.541</td>
    </tr>
    <tr>
      <th>4328180</th>
      <td>21989401003</td>
      <td>6</td>
      <td>272</td>
      <td>1</td>
      <td>654</td>
      <td>1</td>
      <td>20</td>
      <td>6</td>
      <td>1 days 01:00:00</td>
      <td>0.167</td>
    </tr>
  </tbody>
</table>
</div>



```python
sd_data_jan.origin_census_block_group.nunique()  # 219759
sd_data_jan.origin_census_block_group[~sd_data_jan.origin_census_block_group.isin(weatherDF_Jan.geoid)].nunique() # 4869
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[58]: 4869</div>


- Initial Modeling for Jan data only - Part 2
   - Join Jan weather and Jan social distancing dataset together


```python
# sd_data_jan = pd.read_csv("./social_distance_data/jan.csv")
merge_Jan = pd.merge(sd_data_jan.drop('date_time',1), weatherDF_Jan, how='inner', left_on=['origin_census_block_group','day','month'], right_on = ['geoid','day','month'])
merge_Jan.head()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[7]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>ratio_not_leaving</th>
      <th>geoid</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10330210004</td>
      <td>78</td>
      <td>687</td>
      <td>28</td>
      <td>721</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.359</td>
      <td>10330210004</td>
      <td>2019-01-01</td>
      <td>0.400</td>
      <td>99.900</td>
      <td>67.600</td>
      <td>110.300</td>
      <td>43.430</td>
      <td>56.390</td>
      <td>6.263</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10730049022</td>
      <td>259</td>
      <td>5381</td>
      <td>93</td>
      <td>292</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.359</td>
      <td>10730049022</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>94.700</td>
      <td>66.500</td>
      <td>111.900</td>
      <td>50.270</td>
      <td>62.510</td>
      <td>2.908</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11210118001</td>
      <td>91</td>
      <td>1591</td>
      <td>45</td>
      <td>588</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.494</td>
      <td>11210118001</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>98.200</td>
      <td>64.300</td>
      <td>99.600</td>
      <td>49.370</td>
      <td>63.770</td>
      <td>2.908</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11250106021</td>
      <td>392</td>
      <td>4623</td>
      <td>171</td>
      <td>770</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.436</td>
      <td>11250106021</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>94.800</td>
      <td>65.800</td>
      <td>117.700</td>
      <td>50.090</td>
      <td>63.050</td>
      <td>4.026</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40019450014</td>
      <td>97</td>
      <td>615</td>
      <td>33</td>
      <td>204</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.340</td>
      <td>40019450014</td>
      <td>2019-01-01</td>
      <td>0.600</td>
      <td>100.000</td>
      <td>69.300</td>
      <td>87.000</td>
      <td>-3.730</td>
      <td>25.790</td>
      <td>3.803</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>



```python
merge_Jan.info()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 6656671 entries, 0 to 6656670
Data columns (total 19 columns):
 #   Column                        Dtype  
---  ------                        -----  
 0   origin_census_block_group     int64  
 1   device_count                  int64  
 2   distance_traveled_from_home   int64  
 3   completely_home_device_count  int64  
 4   mean_home_dwell_time          int64  
 5   month                         int64  
 6   day                           int64  
 7   dayofweek                     int64  
 8   ratio_not_leaving             float64
 9   geoid                         int64  
 10  date                          object 
 11  precip                        float64
 12  rmax                          float64
 13  rmin                          float64
 14  srad                          float64
 15  tmin                          float64
 16  tmax                          float64
 17  wind_speed                    float64
 18  year                          int32  
dtypes: float64(8), int32(1), int64(9), object(1)
memory usage: 990.3+ MB
</div>



```python
# !pip install openpyxl
narrow_search_list = pd.read_excel("/dbfs/FileStore/tables/CBG_city_in_top_100_yizi.xlsx", usecols = ['GEOID10','NAME','CLASS','ST'],engine='openpyxl')  
print(narrow_search_list.GEOID10.nunique() == narrow_search_list.shape[0])
narrow_search_list.head()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">True
Out[10]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID10</th>
      <th>NAME</th>
      <th>CLASS</th>
      <th>ST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20200001011</td>
      <td>Anchorage</td>
      <td>municipality</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20200001012</td>
      <td>Anchorage</td>
      <td>municipality</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20200001013</td>
      <td>Anchorage</td>
      <td>municipality</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20200001021</td>
      <td>Anchorage</td>
      <td>municipality</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20200001022</td>
      <td>Anchorage</td>
      <td>municipality</td>
      <td>AK</td>
    </tr>
  </tbody>
</table>
</div>



```python
merge_Jan_list = pd.merge(narrow_search_list,merge_Jan.drop('geoid',1), how = 'inner', left_on = ['GEOID10'], right_on = ['origin_census_block_group'])
merge_Jan_list.head()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[11]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID10</th>
      <th>NAME</th>
      <th>CLASS</th>
      <th>ST</th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>ratio_not_leaving</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>95</td>
      <td>6618</td>
      <td>38</td>
      <td>786</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.400</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>98.100</td>
      <td>45.900</td>
      <td>131.500</td>
      <td>24.530</td>
      <td>43.430</td>
      <td>4.474</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>102</td>
      <td>7194</td>
      <td>39</td>
      <td>763</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.382</td>
      <td>2019-01-02</td>
      <td>0.000</td>
      <td>69.500</td>
      <td>25.000</td>
      <td>135.900</td>
      <td>23.810</td>
      <td>46.850</td>
      <td>3.803</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>96</td>
      <td>8033</td>
      <td>31</td>
      <td>710</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0.323</td>
      <td>2019-01-03</td>
      <td>0.000</td>
      <td>53.200</td>
      <td>16.900</td>
      <td>140.200</td>
      <td>30.650</td>
      <td>54.950</td>
      <td>3.132</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>91</td>
      <td>7932</td>
      <td>21</td>
      <td>739</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0.231</td>
      <td>2019-01-04</td>
      <td>0.000</td>
      <td>46.000</td>
      <td>12.700</td>
      <td>140.900</td>
      <td>37.310</td>
      <td>63.770</td>
      <td>3.803</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>98</td>
      <td>7286</td>
      <td>29</td>
      <td>683</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>0.296</td>
      <td>2019-01-05</td>
      <td>18.900</td>
      <td>58.700</td>
      <td>19.700</td>
      <td>98.500</td>
      <td>39.470</td>
      <td>64.310</td>
      <td>7.158</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>



```python
merge_Jan_list.shape
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[12]: (1349693, 22)</div>



```python
merge_Jan_list[merge_Jan_list.GEOID10==40130101022].head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID10</th>
      <th>NAME</th>
      <th>CLASS</th>
      <th>ST</th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>ratio_not_leaving</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>95</td>
      <td>6618</td>
      <td>38</td>
      <td>786</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.400</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>98.100</td>
      <td>45.900</td>
      <td>131.500</td>
      <td>24.530</td>
      <td>43.430</td>
      <td>4.474</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>102</td>
      <td>7194</td>
      <td>39</td>
      <td>763</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.382</td>
      <td>2019-01-02</td>
      <td>0.000</td>
      <td>69.500</td>
      <td>25.000</td>
      <td>135.900</td>
      <td>23.810</td>
      <td>46.850</td>
      <td>3.803</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>96</td>
      <td>8033</td>
      <td>31</td>
      <td>710</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0.323</td>
      <td>2019-01-03</td>
      <td>0.000</td>
      <td>53.200</td>
      <td>16.900</td>
      <td>140.200</td>
      <td>30.650</td>
      <td>54.950</td>
      <td>3.132</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>91</td>
      <td>7932</td>
      <td>21</td>
      <td>739</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0.231</td>
      <td>2019-01-04</td>
      <td>0.000</td>
      <td>46.000</td>
      <td>12.700</td>
      <td>140.900</td>
      <td>37.310</td>
      <td>63.770</td>
      <td>3.803</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40130101022</td>
      <td>Scottsdale</td>
      <td>city</td>
      <td>AZ</td>
      <td>40130101022</td>
      <td>98</td>
      <td>7286</td>
      <td>29</td>
      <td>683</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>0.296</td>
      <td>2019-01-05</td>
      <td>18.900</td>
      <td>58.700</td>
      <td>19.700</td>
      <td>98.500</td>
      <td>39.470</td>
      <td>64.310</td>
      <td>7.158</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>



```python
merge_Jan_list.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID10</th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>ratio_not_leaving</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
      <td>1349693.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>282377492890.544</td>
      <td>282377492890.544</td>
      <td>100.148</td>
      <td>2311.096</td>
      <td>39.430</td>
      <td>658.565</td>
      <td>1.000</td>
      <td>16.001</td>
      <td>2.903</td>
      <td>0.391</td>
      <td>2.634</td>
      <td>80.473</td>
      <td>45.083</td>
      <td>101.881</td>
      <td>31.462</td>
      <td>48.798</td>
      <td>10.005</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>161096236769.261</td>
      <td>161096236769.261</td>
      <td>100.148</td>
      <td>13882.760</td>
      <td>40.144</td>
      <td>115.808</td>
      <td>0.000</td>
      <td>8.944</td>
      <td>1.940</td>
      <td>0.106</td>
      <td>6.712</td>
      <td>16.997</td>
      <td>16.778</td>
      <td>39.213</td>
      <td>14.986</td>
      <td>16.524</td>
      <td>5.143</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40130101022.000</td>
      <td>40130101022.000</td>
      <td>5.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.006</td>
      <td>0.000</td>
      <td>24.900</td>
      <td>4.000</td>
      <td>17.700</td>
      <td>-31.450</td>
      <td>-13.450</td>
      <td>0.895</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>120310167281.000</td>
      <td>120310167281.000</td>
      <td>53.000</td>
      <td>1066.000</td>
      <td>19.000</td>
      <td>599.000</td>
      <td>1.000</td>
      <td>8.000</td>
      <td>1.000</td>
      <td>0.323</td>
      <td>0.000</td>
      <td>69.200</td>
      <td>33.000</td>
      <td>71.100</td>
      <td>23.270</td>
      <td>36.410</td>
      <td>6.263</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>320310017011.000</td>
      <td>320310017011.000</td>
      <td>78.000</td>
      <td>1789.000</td>
      <td>31.000</td>
      <td>666.000</td>
      <td>1.000</td>
      <td>16.000</td>
      <td>3.000</td>
      <td>0.395</td>
      <td>0.000</td>
      <td>82.800</td>
      <td>44.200</td>
      <td>103.300</td>
      <td>32.450</td>
      <td>49.910</td>
      <td>8.948</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>410510006014.000</td>
      <td>410510006014.000</td>
      <td>115.000</td>
      <td>2646.000</td>
      <td>47.000</td>
      <td>731.000</td>
      <td>1.000</td>
      <td>24.000</td>
      <td>5.000</td>
      <td>0.462</td>
      <td>1.200</td>
      <td>96.700</td>
      <td>57.600</td>
      <td>132.200</td>
      <td>42.350</td>
      <td>61.610</td>
      <td>12.751</td>
      <td>2019.000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>550791874001.000</td>
      <td>550791874001.000</td>
      <td>5195.000</td>
      <td>3639693.000</td>
      <td>3080.000</td>
      <td>1364.000</td>
      <td>1.000</td>
      <td>31.000</td>
      <td>6.000</td>
      <td>0.949</td>
      <td>72.700</td>
      <td>100.000</td>
      <td>100.000</td>
      <td>239.100</td>
      <td>74.390</td>
      <td>85.010</td>
      <td>33.330</td>
      <td>2019.000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# no missing value
merge_Jan_list.isna().sum()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[39]: GEOID10                         0
NAME                            0
CLASS                           0
ST                              0
origin_census_block_group       0
device_count                    0
distance_traveled_from_home     0
completely_home_device_count    0
mean_home_dwell_time            0
month                           0
day                             0
dayofweek                       0
ratio_not_leaving               0
date                            0
precip                          0
rmax                            0
rmin                            0
srad                            0
tmin                            0
tmax                            0
wind_speed                      0
year                            0
dtype: int64</div>



```python
corr = merge_Jan_list.iloc[:,4:].corr()
corr.drop(['month','year'],1,inplace=True)
corr.drop(['month','year'],0,inplace=True)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask,0)] = True
print('Correlation Matrix : ')
sns.heatmap(corr, mask = mask,square = True, annot = False , cmap="YlGnBu")

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,10)
```


    
![png](output_53_0.png)
    


- Initial Modeling for Jan data only - Part 3 - Identify an optimal response
   - Response Visualization - 1
     - distance_traveled_from_home -  highly skewed


```python
# # understand predictor  = distance traveled, highly right skewed
plt.hist(merge_Jan_list.distance_traveled_from_home, bins = range(0, 1000000,500), log = True)
plt.show()
```


    
![png](output_55_0.png)
    



```python
Q1 = np.percentile(merge_Jan_list.distance_traveled_from_home, 25, interpolation = 'midpoint') 
Q3 = np.percentile(merge_Jan_list.distance_traveled_from_home, 75, interpolation = 'midpoint')
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR 
print(lower, upper, IQR)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">-1304.0 5016.0 1580.0
</div>



```python
# merge_Jan_list.distance_traveled_from_home[merge_Jan_list.distance_traveled_from_home > 500000].count() # 167
# merge_Jan_list.distance_traveled_from_home[merge_Jan_list.distance_traveled_from_home > 100000].count() #791
# # how many 0 distance, this is important for later transformation, 18743 , not significant
# merge_Jan_list.distance_traveled_from_home[merge_Jan_list.distance_traveled_from_home == 0].count()  # 3699
merge_Jan_list.distance_traveled_from_home[merge_Jan_list.distance_traveled_from_home > 10000].count() / merge_Jan_list.distance_traveled_from_home.count() #  0.005410860099296655
merge_Jan_list.distance_traveled_from_home[merge_Jan_list.distance_traveled_from_home > 5100].count() / merge_Jan_list.distance_traveled_from_home.count()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[77]: 0.0347249337441922</div>



```python
# Try a cut off of 50000 meters
plt.hist(merge_Jan_list.distance_traveled_from_home, bins = range(0, 200000,500), log = True)
plt.show()
```


    
![png](output_58_0.png)
    



```python
# Try a cut off of 10000 meters

plt.hist(merge_Jan_list.distance_traveled_from_home, bins = range(0,5100,100), log = True)
plt.show()
```


    
![png](output_59_0.png)
    


- Initial Modeling for Jan data only - Part 3
   - Response Visualization 2
     - mean_home_dwell_home


```python
plt.hist(merge_Jan_list.mean_home_dwell_time, bins = range(0, 1500,30))
plt.show()# 
```


    
![png](output_61_0.png)
    


- Initial Modeling for Jan data only - Part 3
   - Response Visualization 3
     - ratio of not leaving


```python
plt.hist(merge_Jan_list.ratio_not_leaving)
plt.show()# 
```


    
![png](output_63_0.png)
    


- Initial Modeling for Jan data only - Part 4
   - Visualization of response(mean_distance_traveled_from_home) with features
     - this response variable is not optimal


```python
merge_Jan_1 = merge_Jan_list[merge_Jan_list.distance_traveled_from_home <= 5100]
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
# take a sample for visualization
merge_Jan_1_sample = merge_Jan_1.sample(frac=0.1)

fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_1_sample.groupby(['dayofweek']).median()['distance_traveled_from_home'].plot(ax=ax)
plt.ylabel('median distance travel')

```


    
![png](output_66_0.png)
    



```python
# relationship between distance and exploratory variables

for x in merge_Jan_1_sample.columns[14:21]:
  plt.scatter(x=x, y='distance_traveled_from_home', data=merge_Jan_1_sample)
  plt.xlabel(x)
  plt.ylabel('distance')
  plt.show()

```


    
![png](output_67_0.png)
    



    
![png](output_67_1.png)
    



    
![png](output_67_2.png)
    



    
![png](output_67_3.png)
    



    
![png](output_67_4.png)
    



    
![png](output_67_5.png)
    



    
![png](output_67_6.png)
    


- Initial Modeling for Jan data only - Part 5
   - Visualization of response(mean_home_dwell_time) with features


```python
fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_1_sample.groupby(['dayofweek']).median()['mean_home_dwell_time'].plot(ax=ax)
plt.ylabel('mean_home_dwell_time')
```


    
![png](output_69_0.png)
    



```python
# relationship between distance and exploratory variables

for x in merge_Jan_1_sample.columns[14:21]:
  plt.scatter(x=x, y='mean_home_dwell_time', data=merge_Jan_1_sample)
  plt.xlabel(x)
  plt.ylabel('home dwell time')
  plt.show()
```


    
![png](output_70_0.png)
    



    
![png](output_70_1.png)
    



    
![png](output_70_2.png)
    



    
![png](output_70_3.png)
    



    
![png](output_70_4.png)
    



    
![png](output_70_5.png)
    



    
![png](output_70_6.png)
    


- Initial Modeling for Jan data only - Part 5
   - Visualization of response(ratio of not leaving) with features


```python
# take a sample for visualization
fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_1_sample.groupby(['dayofweek']).median()['ratio_not_leaving'].plot(ax=ax)
plt.ylabel('ratio_not_leaving')
```


    
![png](output_72_0.png)
    



```python
# relationship between distance and exploratory variables

for x in merge_Jan_1_sample.columns[14:21]:
  plt.scatter(x=x, y='ratio_not_leaving', data=merge_Jan_1_sample)
  plt.xlabel(x)
  plt.ylabel('ratio_not_leaving')
  plt.show()
```


    
![png](output_73_0.png)
    



    
![png](output_73_1.png)
    



    
![png](output_73_2.png)
    



    
![png](output_73_3.png)
    



    
![png](output_73_4.png)
    



    
![png](output_73_5.png)
    



    
![png](output_73_6.png)
    



```python
merge_Jan_2 =  merge_Jan_1[merge_Jan_1['NAME'] == 'Chicago']
merge_Jan_2.head()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[21]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID10</th>
      <th>NAME</th>
      <th>CLASS</th>
      <th>ST</th>
      <th>origin_census_block_group</th>
      <th>device_count</th>
      <th>distance_traveled_from_home</th>
      <th>completely_home_device_count</th>
      <th>mean_home_dwell_time</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>ratio_not_leaving</th>
      <th>date</th>
      <th>precip</th>
      <th>rmax</th>
      <th>rmin</th>
      <th>srad</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>370188</th>
      <td>170310101001</td>
      <td>Chicago</td>
      <td>city</td>
      <td>IL</td>
      <td>170310101001</td>
      <td>34</td>
      <td>96</td>
      <td>16</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.471</td>
      <td>2019-01-01</td>
      <td>0.000</td>
      <td>81.100</td>
      <td>77.100</td>
      <td>50.400</td>
      <td>26.690</td>
      <td>31.730</td>
      <td>9.171</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>370189</th>
      <td>170310101001</td>
      <td>Chicago</td>
      <td>city</td>
      <td>IL</td>
      <td>170310101001</td>
      <td>39</td>
      <td>319</td>
      <td>15</td>
      <td>502</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0.385</td>
      <td>2019-01-02</td>
      <td>0.000</td>
      <td>78.700</td>
      <td>70.400</td>
      <td>25.700</td>
      <td>24.350</td>
      <td>30.830</td>
      <td>12.079</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>370190</th>
      <td>170310101001</td>
      <td>Chicago</td>
      <td>city</td>
      <td>IL</td>
      <td>170310101001</td>
      <td>43</td>
      <td>98</td>
      <td>10</td>
      <td>494</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0.233</td>
      <td>2019-01-03</td>
      <td>0.000</td>
      <td>77.100</td>
      <td>48.100</td>
      <td>77.100</td>
      <td>24.710</td>
      <td>39.830</td>
      <td>13.422</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>370191</th>
      <td>170310101001</td>
      <td>Chicago</td>
      <td>city</td>
      <td>IL</td>
      <td>170310101001</td>
      <td>42</td>
      <td>1714</td>
      <td>12</td>
      <td>446</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0.286</td>
      <td>2019-01-04</td>
      <td>0.000</td>
      <td>75.100</td>
      <td>36.100</td>
      <td>77.000</td>
      <td>28.670</td>
      <td>50.450</td>
      <td>8.724</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>370192</th>
      <td>170310101001</td>
      <td>Chicago</td>
      <td>city</td>
      <td>IL</td>
      <td>170310101001</td>
      <td>46</td>
      <td>889</td>
      <td>19</td>
      <td>518</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>0.413</td>
      <td>2019-01-05</td>
      <td>0.000</td>
      <td>76.600</td>
      <td>36.700</td>
      <td>77.800</td>
      <td>31.550</td>
      <td>53.690</td>
      <td>8.948</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>



```python
merge_Jan_2.shape
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[11]: (63982, 22)</div>



```python
# take a sample for visualization
merge_Jan_2_sample = merge_Jan_2.sample(frac=0.1)
fig, ax = plt.subplots(figsize=(5,5))
merge_Jan_2_sample.groupby(['dayofweek']).median()['ratio_not_leaving'].plot(ax=ax)
plt.ylabel('ratio_not_leaving')
```


    
![png](output_76_0.png)
    



```python
for x in merge_Jan_2_sample.columns[14:21]:
  plt.scatter(x=x, y='ratio_not_leaving', data=merge_Jan_2_sample)
  plt.xlabel(x)
  plt.ylabel('ratio_not_leaving')
  plt.show()
```


    
![png](output_77_0.png)
    



    
![png](output_77_1.png)
    



    
![png](output_77_2.png)
    



    
![png](output_77_3.png)
    



    
![png](output_77_4.png)
    



    
![png](output_77_5.png)
    



    
![png](output_77_6.png)
    


- Initial Modeling for Jan data only - Part 6
   - Feature Scaling and Encoding


```python
y = merge_Jan_2['ratio_not_leaving']
sns.distplot(y)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[22]: </div>



    
![png](output_79_1.png)
    



<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f78b5ffcc90&gt;</div>



```python
# # lasso will drop the correlated variables, so we don't need to worry about it
# i dropped the rmin and tmin b/c multicollinearity 
X = merge_Jan_2.iloc[:,np.r_[14:16,17,19:21,11]]
X
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[75]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precip</th>
      <th>rmax</th>
      <th>srad</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>370188</th>
      <td>0.000</td>
      <td>81.100</td>
      <td>50.400</td>
      <td>31.730</td>
      <td>9.171</td>
      <td>1</td>
    </tr>
    <tr>
      <th>370189</th>
      <td>0.000</td>
      <td>78.700</td>
      <td>25.700</td>
      <td>30.830</td>
      <td>12.079</td>
      <td>2</td>
    </tr>
    <tr>
      <th>370190</th>
      <td>0.000</td>
      <td>77.100</td>
      <td>77.100</td>
      <td>39.830</td>
      <td>13.422</td>
      <td>3</td>
    </tr>
    <tr>
      <th>370191</th>
      <td>0.000</td>
      <td>75.100</td>
      <td>77.000</td>
      <td>50.450</td>
      <td>8.724</td>
      <td>4</td>
    </tr>
    <tr>
      <th>370192</th>
      <td>0.000</td>
      <td>76.600</td>
      <td>77.800</td>
      <td>53.690</td>
      <td>8.948</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>436396</th>
      <td>10.000</td>
      <td>100.000</td>
      <td>118.200</td>
      <td>21.110</td>
      <td>10.066</td>
      <td>6</td>
    </tr>
    <tr>
      <th>436397</th>
      <td>2.000</td>
      <td>100.000</td>
      <td>46.500</td>
      <td>33.710</td>
      <td>15.882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>436398</th>
      <td>0.000</td>
      <td>100.000</td>
      <td>94.100</td>
      <td>5.450</td>
      <td>19.014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>436399</th>
      <td>0.000</td>
      <td>95.400</td>
      <td>121.600</td>
      <td>-13.450</td>
      <td>19.238</td>
      <td>2</td>
    </tr>
    <tr>
      <th>436400</th>
      <td>2.300</td>
      <td>100.000</td>
      <td>113.000</td>
      <td>4.190</td>
      <td>9.395</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>63982 rows × 6 columns</p>
</div>



```python
sd_scalar = StandardScaler()
sd_scale =sd_scalar.fit(X.iloc[:,:-1])
X.iloc[:,:-1] = sd_scale.transform(X.iloc[:,:-1])
dayofweek_dic ={0:'Mon',
                1:'Tue',
                2:'Wed',
                3:'Thr',
                4:'Fri',
                5:'Sat',
                6:'Sun'           
               }
X['dayofweek'] = X['dayofweek'].apply(lambda x: dayofweek_dic[x])
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
X = pd.get_dummies(X)
X
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[77]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precip</th>
      <th>rmax</th>
      <th>srad</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>dayofweek_Fri</th>
      <th>dayofweek_Mon</th>
      <th>dayofweek_Sat</th>
      <th>dayofweek_Sun</th>
      <th>dayofweek_Thr</th>
      <th>dayofweek_Tue</th>
      <th>dayofweek_Wed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>370188</th>
      <td>-0.568</td>
      <td>-0.123</td>
      <td>-0.778</td>
      <td>0.179</td>
      <td>-0.641</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>370189</th>
      <td>-0.568</td>
      <td>-0.284</td>
      <td>-1.880</td>
      <td>0.115</td>
      <td>0.046</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>370190</th>
      <td>-0.568</td>
      <td>-0.392</td>
      <td>0.414</td>
      <td>0.750</td>
      <td>0.363</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>370191</th>
      <td>-0.568</td>
      <td>-0.527</td>
      <td>0.409</td>
      <td>1.498</td>
      <td>-0.747</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>370192</th>
      <td>-0.568</td>
      <td>-0.426</td>
      <td>0.445</td>
      <td>1.727</td>
      <td>-0.694</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>436396</th>
      <td>2.351</td>
      <td>1.151</td>
      <td>2.248</td>
      <td>-0.570</td>
      <td>-0.430</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>436397</th>
      <td>0.016</td>
      <td>1.151</td>
      <td>-0.952</td>
      <td>0.318</td>
      <td>0.944</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>436398</th>
      <td>-0.568</td>
      <td>1.151</td>
      <td>1.172</td>
      <td>-1.674</td>
      <td>1.684</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>436399</th>
      <td>-0.568</td>
      <td>0.841</td>
      <td>2.400</td>
      <td>-3.007</td>
      <td>1.737</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>436400</th>
      <td>0.103</td>
      <td>1.151</td>
      <td>2.016</td>
      <td>-1.763</td>
      <td>-0.588</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>63982 rows × 12 columns</p>
</div>



```python
# from sklearn.decomposition import PCA
# pca = PCA(n_components =3)
# pca.fit(X)
# X_pca = pca.transform(X)
# X_pca
# pca.explained_variance_
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[46]: array([1.63406446, 1.48822995, 0.91400446])</div>



```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
rlf = linear_model.Ridge()

grid = GridSearchCV(estimator=rlf, param_grid=dict(alpha=np.array([10, 1,0.1,0.01,0.001,0.0001,0])))
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_estimator_.alpha)

rlf_final = linear_model.Ridge(alpha = 1)
rlf_final.fit(X_train,y_train)
rlf_final_pred = rlf_final.predict(X_train)
rlf_final_pred_test = rlf_final.predict(X_test)

rlf_final_r2_score = r2_score(y_train,rlf_final_pred)
rlf_final_r2_score_test = r2_score(y_test,rlf_final_pred_test)
#pca 9.6%
print(rlf_final_r2_score,rlf_final_r2_score_test)  
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">GridSearchCV(cv=None, error_score=nan,
             estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,
                             max_iter=None, normalize=False, random_state=None,
                             solver=&#39;auto&#39;, tol=0.001),
             iid=&#39;deprecated&#39;, n_jobs=None,
             param_grid={&#39;alpha&#39;: array([1.e+01, 1.e+00, 1.e-01, 1.e-02, 1.e-03, 1.e-04, 0.e+00])},
             pre_dispatch=&#39;2*n_jobs&#39;, refit=True, return_train_score=False,
             scoring=None, verbose=0)
10.0
0.13982850595840823 0.13543623906875257
</div>



```python
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
pol_pred = pol_reg.predict(X_poly)
pol_pred_test = pol_reg.predict(X_poly_test)

pol_r2_score = r2_score(y_train,pol_pred)
pol_r2_score_test = r2_score(y_test,pol_pred_test)
# 18%
print(pol_r2_score,pol_r2_score_test)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">0.3145492774855494 0.29631197041268975
</div>



```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)
rfModel = rf.fit(X_train, y_train)
rf_pred = rfModel.predict(X_train)
rf_r2_score = r2_score(y_train,rf_pred)
rf_pred_test = rfModel.predict(X_test)
rf_r2_score_test = r2_score(y_test,rf_pred_test)
print(rf_r2_score,rf_r2_score_test)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">0.3878847810456796 0.34460560559878406
</div>



```python
# # # Improve random forest
# rf_parameter_grid = {'n_estimators': [100,105,120]
#                      ,'max_depth': list(np.linspace(10, 30, 5, endpoint=True))     
#                      }

# rf1 = RandomForestRegressor(random_state = 42)
# grid_rf1 = GridSearchCV(rf1, rf_parameter_grid, cv=10)
# grid_rf1.fit(X_train, y_train)
# print(grid.best_score_)
# print(grid_rf1.best_estimator_)
# rf_pred_ = grid_rf1.best_estimator_.predict(X_train)
# rf_r2_score_ = r2_score(y_train,rf_pred_)
# print(rf_r2_score_)

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
#  PCA(reduce features, normalize distance, mean group by geoid, max = max(by state), ratio normalize to run, preferable 40-505)
importances = pd.Series(rfModel.feature_importances_, index = X.columns)
importances.nlargest(5).plot(kind='barh').invert_yaxis()
```


    
![png](output_89_0.png)
    



```python
y1 = merge_Jan_1['ratio_not_leaving']
sns.distplot(y1)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[83]: </div>



    
![png](output_90_1.png)
    



<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f78b656fe90&gt;</div>



```python
# # lasso will drop the correlated variables, so we don't need to worry about it
# i dropped the rmin and tmin b/c multicollinearity 
X1 = merge_Jan_1.iloc[:,np.r_[14:16,17,19:21,11]]
X1
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[84]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precip</th>
      <th>rmax</th>
      <th>srad</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>dayofweek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>1.000</td>
      <td>73.900</td>
      <td>101.200</td>
      <td>63.950</td>
      <td>5.592</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.300</td>
      <td>99.300</td>
      <td>67.400</td>
      <td>58.550</td>
      <td>8.500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.000</td>
      <td>96.100</td>
      <td>115.000</td>
      <td>63.590</td>
      <td>2.684</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.000</td>
      <td>42.600</td>
      <td>158.000</td>
      <td>59.450</td>
      <td>3.803</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.000</td>
      <td>40.900</td>
      <td>162.200</td>
      <td>72.230</td>
      <td>2.684</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1349688</th>
      <td>0.000</td>
      <td>80.900</td>
      <td>131.000</td>
      <td>48.290</td>
      <td>9.619</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1349689</th>
      <td>0.000</td>
      <td>60.800</td>
      <td>140.900</td>
      <td>38.390</td>
      <td>7.158</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1349690</th>
      <td>4.500</td>
      <td>100.000</td>
      <td>47.600</td>
      <td>40.010</td>
      <td>11.632</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1349691</th>
      <td>0.000</td>
      <td>75.400</td>
      <td>126.100</td>
      <td>36.950</td>
      <td>16.330</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1349692</th>
      <td>0.000</td>
      <td>41.400</td>
      <td>152.700</td>
      <td>23.810</td>
      <td>6.487</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>1302825 rows × 6 columns</p>
</div>



```python
sd_scalar = StandardScaler()
sd_scale =sd_scalar.fit(X1.iloc[:,:-1])
X1.iloc[:,:-1] = sd_scale.transform(X1.iloc[:,:-1])
dayofweek_dic ={0:'Mon',
                1:'Tue',
                2:'Wed',
                3:'Thr',
                4:'Fri',
                5:'Sat',
                6:'Sun'           
               }
X1['dayofweek'] = X1['dayofweek'].apply(lambda x: dayofweek_dic[x])
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
X1 = pd.get_dummies(X1)
X1
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">Out[86]: </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precip</th>
      <th>rmax</th>
      <th>srad</th>
      <th>tmax</th>
      <th>wind_speed</th>
      <th>dayofweek_Fri</th>
      <th>dayofweek_Mon</th>
      <th>dayofweek_Sat</th>
      <th>dayofweek_Sun</th>
      <th>dayofweek_Thr</th>
      <th>dayofweek_Tue</th>
      <th>dayofweek_Wed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>-0.244</td>
      <td>-0.383</td>
      <td>-0.018</td>
      <td>0.917</td>
      <td>-0.859</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.992</td>
      <td>1.110</td>
      <td>-0.880</td>
      <td>0.590</td>
      <td>-0.295</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.393</td>
      <td>0.922</td>
      <td>0.334</td>
      <td>0.895</td>
      <td>-1.424</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.393</td>
      <td>-2.223</td>
      <td>1.429</td>
      <td>0.644</td>
      <td>-1.207</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.393</td>
      <td>-2.323</td>
      <td>1.536</td>
      <td>1.419</td>
      <td>-1.424</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1349688</th>
      <td>-0.393</td>
      <td>0.028</td>
      <td>0.741</td>
      <td>-0.032</td>
      <td>-0.078</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1349689</th>
      <td>-0.393</td>
      <td>-1.153</td>
      <td>0.994</td>
      <td>-0.633</td>
      <td>-0.556</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1349690</th>
      <td>0.277</td>
      <td>1.151</td>
      <td>-1.384</td>
      <td>-0.535</td>
      <td>0.313</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1349691</th>
      <td>-0.393</td>
      <td>-0.295</td>
      <td>0.616</td>
      <td>-0.720</td>
      <td>1.224</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1349692</th>
      <td>-0.393</td>
      <td>-2.294</td>
      <td>1.294</td>
      <td>-1.517</td>
      <td>-0.686</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1302825 rows × 12 columns</p>
</div>



```python
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size = 0.2)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>



```python
from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor(random_state = 42)
rfModel = rf1.fit(X1_train, y1_train)
rf1_pred = rfModel.predict(X1_train)
rf1_r2_score = r2_score(y1_train,rf1_pred)
rf1_pred_test = rfModel.predict(X1_test)
rf1_r2_score_test = r2_score(y1_test,rf1_pred_test)
print(rf1_r2_score,rf1_r2_score_test)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">0.4239697680856348 0.3362159196493033
</div>



```python
# # # Improve random forest
rf_parameter_grid = {'n_estimators': [100,105,120]
                     ,'max_depth': list(np.linspace(10, 30, 5, endpoint=True))     
                     }

rf2 = RandomForestRegressor(random_state = 42)
grid_rf2 = GridSearchCV(rf2, rf_parameter_grid, cv=10)
grid_rf2.fit(X_train, y_train)
print(grid.best_score_)
print(grid_rf2.best_estimator_)
rf_pred_ = grid_rf2.best_estimator_.predict(X_train)
rf_r2_score_ = r2_score(y_train,rf_pred_)
print(rf_r2_score_)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">0.13935896394681255
RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion=&#39;mse&#39;,
                      max_depth=15.0, max_features=&#39;auto&#39;, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=120, n_jobs=None, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)
0.3814781943490182
</div>



```python

```
