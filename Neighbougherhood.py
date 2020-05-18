# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:30:04 2020

@author: NabilHassan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:03:47 2020

@author: NabilHassan
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim 
import requests
import webbrowser
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from geopandas.tools import geocode
from sklearn.metrics import silhouette_score
df=pd.read_csv("Abu Dhabi Cities.csv")
#CLIENT_ID = 'OC2Z3BAX5LIQGUSSI2ZOZCPB324IIAFRYLG0H1T2OKMKGBNL' # your Foursquare ID
CLIENT_ID = '0HH2B0MRFB2FALD3CL3SQAGF5KPCVO53DS5OEOKOP4MWUCJO'
#CLIENT_SECRET = 'A3N4IZM5ESAKKLGJMXZKRZRA4RDCHHDLPBP0OXXXV5C52MW1' # your Foursquare Secret
CLIENT_SECRET = 'D5KMPZK1RAFC0RSUS3VCUOIAIIA2KVCOWHIP1RJX3D1L0UQS'
VERSION = '20200504'
LIMIT = 10000
radius = 1000
cat="4d4b7105d754a06374d81259"

rest_type =['Gluten-free Restaurant', 'Lebanese Restaurant','Yemeni Restaurant','Asian Restaurant', 
       'Chinese Restaurant', 'Indian Restaurant','Japanese Restaurant', 'Middle Eastern Restaurant',
       'Filipino Restaurant', 'Pakistani Restaurant','Seafood Restaurant', 
       'Mediterranean Restaurant','Italian Restaurant','Vegetarian / Vegan Restaurant',  'French Restaurant',
       'African Restaurant', 'Korean Restaurant', 'Eastern European Restaurant', 
        'Turkish Restaurant', 'German Restaurant', 'American Restaurant', 'Greek Restaurant',
       'South Indian Restaurant','Peruvian Restaurant',  'Thai Restaurant',
        'Russian Restaurant','Mexican Restaurant', 'North Indian Restaurant',
        'Egyptian Restaurant', 'Iraqi Restaurant','Indian Chinese Restaurant', 
       'Vietnamese Restaurant', 'Portuguese Restaurant', 'New American Restaurant',
       'Malay Restaurant', 'English Restaurant','Afghan Restaurant', 'Halal Restaurant', 'Ethiopian Restaurant',
       'Moroccan Restaurant', 'Spanish Restaurant', 'Modern European Restaurant']


geo=geocode(df['Address'], provider='nominatim')



 
nearby_venues_list=pd.DataFrame([])
nearby_venues_list1=pd.DataFrame([])
 
df['lattitude'] =geo.centroid.x
df['longtitude'] =geo.centroid.y
print('test')
 
 
for index, row in df.iterrows():
    url = 'https://api.foursquare.com/v2/venues/search?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&categoryId={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    row['longtitude'], 
    row['lattitude'], 
    radius, 
    LIMIT,cat)
    #print(url)
    results = requests.get(url).json()
    print(row['Place'])
   
    venues = results['response']['venues']
    nearby_venues=json_normalize(venues)
    nearby_venues['Place'] =row['Place']
    nearby_venues_list=nearby_venues_list.append(nearby_venues)
    
 
Cate_div=nearby_venues_list["categories"].str.get(0)
 
Categories_Details=Cate_div.apply(pd.Series )

Categories_Details.rename(columns={'name':'categoryname'},inplace=True)
 
nearby_venues_lists= pd.concat([nearby_venues_list, Categories_Details], axis=1, sort=False)

nearby_venues_lists= nearby_venues_lists[nearby_venues_lists.categoryname.isin(rest_type)]

df=df[df.Place.isin(nearby_venues_lists['Place'] )]

vnc_onehot = pd.get_dummies(nearby_venues_lists[['categoryname']], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
vnc_onehot['Neighbourhood'] =nearby_venues_lists['Place'] 

# move neighborhood column to the first column
fixed_columns = [vnc_onehot.columns[-1]] + list(vnc_onehot.columns[:-1])
vnc_onehot = vnc_onehot[fixed_columns]

vnc_onehot.head()
vnc_ws_grouped = vnc_onehot.groupby('Neighbourhood').sum().reset_index()
vnc_ws_grouped


num_top_venues = 10

for hood in vnc_ws_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = vnc_ws_grouped[vnc_ws_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
 
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]
    


indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = vnc_ws_grouped['Neighbourhood']

for ind in np.arange(vnc_ws_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(vnc_ws_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()



kclusters = 5

vnc_grouped_clustering = vnc_ws_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(vnc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
vancouver_merged = df.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Place')

vancouver_merged.head()


distortions = []
K = range(2,8)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(vnc_grouped_clustering)
    kmeanModel.fit(vnc_grouped_clustering)
    distortions.append(sum(np.min(cdist(vnc_grouped_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vnc_grouped_clustering.shape[0])
    label = kmeanModel.labels_
    sil_coeff = silhouette_score(vnc_grouped_clustering, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]


address = 'Abu Dhabi ,United Arab Emirates'
location = geocode(address,provider='nominatim')
latitude = location.centroid.x
longitude = location.centroid.y
map_clusters = folium.Map(location=[longitude,latitude], zoom_start=12)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(vancouver_merged['longtitude'],vancouver_merged['lattitude'],vancouver_merged['Place'], vancouver_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters
map_clusters.save("mymap.html")
webbrowser.open('mymap.html')