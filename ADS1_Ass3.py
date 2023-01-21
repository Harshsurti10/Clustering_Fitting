"""
Starting Clustering Part
"""
# importing necessary modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet

# function to read file
def readFile(x):
    '''
        Reading CSV file from this function in original form
        
        x : csv filename
    
        Data : variable for storing csv file

    '''
    Data = pd.read_csv("Ele_Cos.csv");
    Data = pd.read_csv(x)
    Data = Data.fillna(0.0)
    return Data
 
# calling function to show dataframe 
Data = readFile("Ele_Cos.csv")

print("\nElecticity Consumption: \n", Data)

# dropping certain columns which are donot need to clean data
Data = Data.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960'], axis=1)

print("\nElecticity Consumption: \n", Data)

# transpose dataframe
Data = pd.DataFrame.transpose(Data)

print("\nTransposed Electicity Consumption: \n",Data)

# Creating header with header information
header1 = Data.iloc[0].values.tolist()
Data.columns = header1

print("\nElecticity Consumption Header: \n", Data)

# removing two rows from dataframe
Data = Data.iloc[2:]

print("\nElecticity Consumption after choosing certain rows: \n", Data)

# creating a dataframe for two columns to store original values
ele_con = Data[["Sri Lanka","Spain"]].copy()

# extracting maximum and minmum value from new dataframe
max_val = ele_con.max()

min_val = ele_con.min()

ele_con = (ele_con - min_val) / (max_val - min_val) # operation of minimun and maximum values

print("\nMin and Max operation on Electicity Consumption: \n", ele_con)

# set up clusterer and number of clusters
ncluster = 5

kmeans = cluster.KMeans(n_clusters=ncluster)

# fitting the data where the results are stored in kmeans object
kmeans.fit(ele_con)

labels = kmeans.labels_ # labels is number of associated clusters

# extracting estimated cluster centres
cen = kmeans.cluster_centers_
print("\nCluster Centres: \n", cen)
# calculating the silhoutte score
print("\nSilhoutte Score: \n",skmet.silhouette_score(ele_con, labels))

# plot using the labels to select any colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:brown", "tab:green", "tab:blue", "tab:red", "tab:olive"]
       
# loop over the different labels    
for l in range(ncluster): 
    plt.plot(ele_con[labels==l]["Sri Lanka"], ele_con[labels==l]["Spain"], 
             marker="o", markersize=3, color=col[l])

# display cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Sri Lanka")

plt.ylabel("Spain")

plt.show()

print("\nCentres: \n", cen)


df_cen = pd.DataFrame(cen, columns=["Sri Lanka", "Spain"])

print(df_cen)

df_cen = df_cen * (max_val - min_val) + max_val

ele_con = ele_con * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print("\nDataframe Centre: \n", df_cen)

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:gray", "tab:olive"]


for l in range(ncluster): # loop over the different labels
    plt.plot(ele_con[labels==l]["Sri Lanka"], ele_con[labels==l]["Spain"], "o", markersize=3, color=col[l])
    

# show cluster centres
plt.plot(df_cen["Sri Lanka"], df_cen["Spain"], "dk", markersize=10)

plt.xlabel("Sri Lanka")

plt.ylabel("Spain")

plt.title("Electricity Consumption(KWH per Capita)")

plt.show()

print("\nCentres: \n", cen)

"""
CURVE FIT PART STARTS
"""
# importing necessary modules
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# function to read file
def readFile(y):
    '''
        Reading CSV file through funcation in original form
        
        x : csv filename
    
        Returns
        annual_pop : variable for storing csv file

    '''
    annual_pop = pd.read_csv("An_population.csv");
    annual_pop = pd.read_csv(y)
    annual_pop = annual_pop.fillna(0.0)
    return annual_pop


# calling readFile function to display dataframe 
annual_pop = readFile("An_population.csv")

print("\nAnnual Population: \n", annual_pop)
print(annual_pop)

print(annual_pop)

# transpose dataframe
annual_pop = annual_pop.transpose()

print("\nTransposed Annual Population: \n", annual_pop)

# populating header with header information
header2 = annual_pop.iloc[0].values.tolist()

annual_pop.columns = header2

print("\nAnnual Population Header: \n", annual_pop)

# select particular column
annual_pop = annual_pop["Mali"]

print("\nAnnual Population after choosing certain column: \n", annual_pop)

# rename column
annual_pop.columns = ["Population:"]

print("\nRenamed Annual Population: \n", annual_pop)


# extracting particular rows
annual_pop = annual_pop.iloc[5:]

annual_pop = annual_pop.iloc[:-1]

print("\nAnnual Population after selecting particular rows: \n", annual_pop)

# resetn index of dataframe
annual_pop = annual_pop.reset_index()

print("\nAnnual Population Growth reset index: \n", annual_pop)


# rename columns
annual_pop = annual_pop.rename(columns={"index": "Year", "Mali": "Population"} )

print("\nAnnual Population Growth after renamed columns: \n", annual_pop)

print(annual_pop.columns)

# plot line graph
annual_pop.plot("Year", "Population", label="Population")

plt.legend()

plt.title("Population Growth")

plt.show()

# curve fit with exponential function
def exponential(s, q0, h):
    '''
        Calculates exponential function with scale factor.
    '''
    s = s - 1960.0
    x = q0 * np.exp(h*s)
    return x


# performing best fit in curve fit
print(type(annual_pop["Year"].iloc[1]))

annual_pop["Year"] = pd.to_numeric(annual_pop["Year"])

print("\nGDP Growth Type: \n", type(annual_pop["Year"].iloc[1]))

param, covar = opt.curve_fit(exponential, annual_pop["Year"], annual_pop["Population"],
p0=(1.364463, 0.03))

# plotting best fit
annual_pop["fit"] = exponential(annual_pop["Year"], *param)

annual_pop.plot("Year", ["Population", "fit"], label=["Population", "Fit"])

plt.legend()

plt.title("Annual Population")

plt.show()

# predict fit for future years
year = np.arange(1960, 2031)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()

plt.plot(annual_pop["Year"], annual_pop["Population"], label="Population")

plt.plot(year, forecast, label="Forecast")

plt.xlabel("Year")

plt.ylabel("Population")

plt.title("Annual Population")

plt.legend()

plt.show()

# err_ranges function
def err_ranges(x, exponential, param, sigma):
    '''
        calculates the function, parameter, and sigma upper and lower bounds for a single value or array x. 
        All combinations of +/- sigma's function values are computed, and the lowest and maximum values are established.
        Can be used for all parameter counts and sigmas greater than 1.
    '''
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper