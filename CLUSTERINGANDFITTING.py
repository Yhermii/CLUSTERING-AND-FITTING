import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from sklearn.datasets import make_blobs
import sklearn.preprocessing as pp
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
from errors import error_prop, covar_to_corr

def read_and_drop_columns(file_path):
    """Read the CSV file into a DataFrame and drops Country Code 
    and Series Code column 
    """
    df_clust = pd.read_csv(file_path)

    # Drop specified columns from the DataFrame
    df_clust = df_clust.drop(columns=["Country Code", "Series Code"])

    # Return the modified DataFrame
    return df_clust

file_path = '/Users/babayhermi/Downloads/clustering.csv'
df_clust = read_and_drop_columns(file_path)
print(df_clust.head())

# filtering china out of the dataframe
df_china = df_clust[df_clust['Country Name'] == 'China']
print(df_china)

# Drop columns ("Country Code" and "Series Code") from the dataframe
#df_china = df_china.drop(columns= ["Country Code", "Series Code"])
print(df_china.info())


#clean data 
df_tclus = df_clust.transpose() 
df_tclus.columns = df_tclus.iloc[0]
df_tclus = df_tclus[4:-1]
df_tclus.index.names = ["Year"]
print(df_tclus.head())
(df_tclus.describe())


# Using 1990 and 2020 for clustering.
df_clust = df_clust[df_clust["1990"].notna() & df_clust["2020"].notna()]
df_clust = df_clust.reset_index(drop=True)

# extract 1990
growth = df_clust[["Country Name", "1990"]].copy()
print(growth)

# and calculate the growth over 30 years
growth["Growth"] = 100.0/30.0 * (df_clust["2020"]-df_clust["1990"]) / df_clust["1990"] 
print(growth.describe())

#plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(growth["1990"], growth["Growth"])
plt.xlabel("Total green house gas 1990")
plt.ylabel("Growth per year [%]")
plt.title("CLUSTER FOR GHG GROWTH/YR(%) AND TOTAL GHG 1990")
plt.show()

# create a scaler object
scaler = pp.RobustScaler()

# set up the scaler
# extract the columns for clustering
df_extract = growth[["1990", "Growth"]]
scaler.fit(df_extract)

# apply the scaling
normalize = scaler.transform(df_extract)

plt.figure(figsize=(8, 8))
plt.scatter(normalize[:, 0], normalize[:, 1])

plt.xlabel("Total green house gas 1990")
plt.ylabel("Growth per year [%]")
plt.title("NORMALIZED CLUSTER FOR GHG GROWTH/YR(%) AND TOTAL GHG 1990")
plt.show()

def compute_silhouette_score(data, num_clusters):
    """Calculates silhouette score for a given number of clusters."""
    
    # Set up the KMeans clusterer with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    
    # Fit the data, and store the results in the kmeans object
    kmeans.fit(data)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Calculate the silhouette score
    score = silhouette_score(data, labels)
    
    return score

# Print the number of samples in your data
num_samples = df_extract.shape[0]
print(f"Number of samples in your data: {num_samples}")

# Ensure '2020' is not in the index
df_extract.reset_index(drop=True, inplace=True)

# Check if '2020' is in the columns
if '2020' in df_extract.columns:
    # Calculate silhouette score for 2 to min(10, num_samples - 1) clusters
    for num_clusters in range(2, min(11, num_samples + 1)):
        score = compute_silhouette_score(df_extract[['1990', '2020']], num_clusters)
        print(f"The silhouette score for {num_clusters: 3d} clusters is {score: 7.4f}")
else:
    print("Column '2020' not found in DataFrame.")


# Kmeans clustering
#df_clust, labels = make_blobs(n_samples=300, centers=3, random_state=42)

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(normalize)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_
print(labels)
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]    


# print(labels)
print(df_extract)

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(df_extract["1990"],df_extract["Growth"], 10, labels, marker="o", cmap='viridis')
    
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
plt.xlabel("Total green house gas 1990")
plt.ylabel("Growth per year [%]")
plt.title("K-MEANS CLUSTER FOR GHG GROWTH/YR(%) AND TOTAL GHG 1990")

plt.show()

growth['labels'] = labels
print(growth)



# Data for China
years = np.arange(1990, 2021)
emissions = np.array([
    3238858.656, 3386235.935, 3515192.811, 3744912.203, 3904475.172, 4307873.902,
    4328947.13, 4324409.747, 4442509.102, 4375022.621, 4567274.433, 4759814.089,
    5053060.571, 5724514.257, 6489937.932, 7263558.578, 7942463.982, 8551053.498,
    8805331.438, 9380446.702, 10211636.79, 11089792.42, 11374795.58, 11861843.52,
    11940737.42, 11804696.79, 11773340.79, 12014198.87, 12524335.16, 12732245.13,
    12942868.34
])

# EXPONENTIAL FUNCTION FOR FITTING
# Define the exponential growth model function
def exponential_growth_model(x, a, b):
    return a * np.exp(b * (x - years[0]))

# Fit the model to the data
params, covariance = curve_fit(exponential_growth_model, years, emissions)

# Extract the fitted parameters
a, b = params

# Predict emissions for the next 20 years
future_years = (2030, 2040)
predicted_emissions = exponential_growth_model(future_years, a, b)

# Calculate confidence intervals
sigma = np.sqrt(np.diag(covariance))
confidence_intervals = 1.96 * sigma  # 95% confidence interval

# Plot the original data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(years, emissions, label='Actual Data')
plt.plot(years, exponential_growth_model(years, a, b), label='Fitted Model', color='red')
plt.plot(future_years, predicted_emissions, label='Predicted Values', linestyle='dashed', color='green')
plt.fill_between(future_years, predicted_emissions - confidence_intervals[0], predicted_emissions + confidence_intervals[0], color='green', alpha=0.2)

plt.xlabel('Year')
plt.ylabel('Total Greenhouse Gas Emissions (kt of CO2 equivalent)')
plt.title('Exponential Growth Model for China\'s Greenhouse Gas Emissions')
plt.legend()
plt.show()

# Display predicted values for the next 20 years with confidence intervals
for year, value, interval in zip(future_years, predicted_emissions, confidence_intervals):
    print(f"Year {year}: Predicted Emissions = {value:.2f} kt (± {interval:.2f} kt)")


# CONFIDENCE INTERVAL
# Calculate confidence intervals using the provided error_prop function
confidence_intervals = error_prop(years, exponential_growth_model, params, covariance)

# Convert covariance matrix to correlation matrix
correlation_matrix = covar_to_corr(covariance)

# Predict emissions for the next 20 years
future_years = (2030, 2040)
predicted_emissions = exponential_growth_model(future_years, a, b)

# Calculate confidence intervals for the predicted values
predicted_intervals = error_prop(future_years, exponential_growth_model, params, correlation_matrix)

# Plot the original data, the fitted model, and the confidence intervals
plt.figure(figsize=(10, 6))
plt.scatter(years, emissions, label='Actual Data')
plt.plot(years, exponential_growth_model(years, a, b), label='Fitted Model', color='red')
plt.plot(future_years, predicted_emissions, label='Predicted Values', linestyle='dashed', color='green')
plt.fill_between(future_years, predicted_emissions - predicted_intervals, predicted_emissions + predicted_intervals, color='green', alpha=0.2)

plt.xlabel('Year')
plt.ylabel('Total Greenhouse Gas Emissions (kt of CO2 equivalent)')
plt.title('Exponential Growth Model for China\'s Greenhouse Gas Emissions with Confidence Intervals')
plt.legend(loc='upper left')
plt.show()

# Display predicted values for the next 20 years with confidence intervals
for year, value, interval in zip(future_years, predicted_emissions, predicted_intervals):
    print(f"Year {year}: Predicted Emissions = {value:.2f} kt (± {interval:.2f} kt)")