# %%
!pip install pandas matplotlib seaborn scikit-learn
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.neighbors import KernelDensity
pd.options.display.max_columns = None
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_validate 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge  , LogisticRegression , QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor , HistGradientBoostingRegressor 
from sklearn.metrics import mean_absolute_percentage_error , r2_score
from sklearn.model_selection import train_test_split, cross_validate
# %%
df_test=pd.read_csv('test.csv')
# %%
df_test
# %%
# df.drop(columns=["Unnamed: 0"],inplace=True)
# %% [markdown]
# ## Description preprocessing
# %%
df_test['description'].fillna('', inplace=True)
# %%
df_test['description'].fillna('', inplace=True)
# %%
df_test['description']=df_test.description.str.replace("آسانسور ندارد"," ")
df_test['description']=df_test.description.str.replace("سانسور ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("آسانسور ندارد"," ")
df_test['description']=df_test.description.str.replace("سانسور ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("پارکینگ ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("پارکینگ ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("انباری ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("انباری ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("تراس ندارد"," ")
# %%
df_test['description']=df_test.description.str.replace("تراس ندارد"," ")
# %%
df_test.columns
# %%
df_test.drop(columns=['advantage','anbari','parking','asansor'],inplace=True)
# %%
df_test.drop('price_meter',axis=1,inplace=True)
# %%
def print_random_des(feature):
    i=np.random.randint(len(df_test))
    print(df_test.iloc[i][feature])
    return i
print_random_des('description')

# %%
import re

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', data)

def remove_parentheses(text):
    return re.sub(r'[()]', ' ',text)

def remove_links(data):
    return re.sub("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)", ' ', data)

def normalizing_lines(text):          
    return re.sub(r'(\n)+', r' ',text)

def normalizing_punctuations(text):
    text = re.sub('[`~@#$%^&*\-_=+/\\|\[\]<>{}\'«»"]', ' ', text)
    text=text.strip()
    return re.sub(r'([:!?,;،.])+', r' ', text)

def normalize_whitespaces(data):
    data = re.sub("\n+", " ", data)
    data = re.sub("  +", ' ', data)
    data = re.sub("\t+", " ", data)
    data = re.sub("\r+", " ", data)
    data = re.sub("\v+", " ", data)
    data = re.sub("\f+", " ", data)
    return data

def preprocessing_pipeline(text):
    text=remove_emojis(text)
    text=remove_links(text)
    text=remove_parentheses(text)
    text=normalizing_lines(text)
    text=normalize_whitespaces(text)
    text=normalizing_punctuations(text)
    return text
# %%
#not_nan_val=df['description'].isna().values
df_test['description_cleaned']=df_test['description'].apply(preprocessing_pipeline)
# %%
df_test['description_cleaned']=df_test['description'].apply(preprocessing_pipeline)
# %%
print_random_des('description_cleaned')
# %% [markdown]
# ### hazm
# 
# %%
from hazm import *
# %%
normalizer=Normalizer()
df['description_cleaned']=df['description_cleaned'].apply(normalizer.normalize)
# %%
df_test['description_cleaned']=df_test['description_cleaned'].apply(normalizer.normalize)
# %%
print_random_des('description_cleaned')

# %%
df['final_description']=df['description_cleaned'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopwords_list()]))
# %%
df_test['final_description']=df_test['description_cleaned'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopwords_list()]))
# %%
df.to_csv('final_cvs')
# %%
df_test.to_csv('test_final_cvs')
# %%
print_random_des('final_description')

# %%
print(df['description'].iloc[25938])
# %% [markdown]
# # import preprocessed dataFrame 
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.neighbors import KernelDensity
pd.options.display.max_columns = None
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_validate 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge  , LogisticRegression , QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor , HistGradientBoostingRegressor 
from sklearn.metrics import mean_absolute_percentage_error , r2_score
from sklearn.model_selection import train_test_split, cross_validate
# %%
df=pd.read_csv('final_cvs.csv')
df.drop(columns=["Unnamed: 0"],inplace=True)
df_test=pd.read_csv('test_final_cvs.csv')
df_test.drop(columns=["Unnamed: 0"],inplace=True)
# %% [markdown]
# ### Make Graph
# %%
df
# %%
import plotly.express as px
px.scatter(df , x="longitude", y="latitude", hover_name="neighborhood", width=400, height=400)
# %%
df.columns
# %%
df.info()
# %%
df.describe()
# %%
df['type']=df['type'].astype('int8')
df['floor']=df['floor'].astype('int8')
df['bedrooms']=df['bedrooms'].astype('int16')
df['sanad']=df['sanad'].astype('int8')

# %%
shuffle_indices = np.random.choice(np.arange(df.shape[0]), size=100, replace=False)
df_shuffled = df.iloc[shuffle_indices].reset_index(drop=True)
# %%
df_shuffled
# %%
import pandas as pd
import networkx as nx
from geopy.distance import geodesic

G = nx.Graph()

for index, row in df_shuffled.iterrows():
    G.add_node(index, pos=(row['latitude'], row['longitude']), Type=row['type'], area=row['area'], neighborhood=row['neighborhood'] ,
               description=row['description'] , age=row['age'] , floor=row['floor'] , bedrooms=row['bedrooms'] ,price=row['price'] ,
              region=row['region'] , sanad=row['sanad'] , final_description=row['final_description'] )


# %%
from scipy.spatial import cKDTree

coords = np.radians(df_shuffled[['latitude', 'longitude']].values)
tree = cKDTree(coords)

radius = 1 / 6371.0

pairs = tree.query_pairs(radius)
# %%
from tqdm import tqdm

for i, j in tqdm(pairs):
    coord1 = (df_shuffled.loc[i, 'latitude'], df_shuffled.loc[i, 'longitude'])
    coord2 = (df_shuffled.loc[j, 'latitude'], df_shuffled.loc[j, 'longitude'])
    distance = geodesic(coord1, coord2).kilometers
    G.add_edge(i, j, weight=distance)
# %%
for node in G.nodes(data=True):
    print(f"Node {node[0]}: {node[1]}")

for edge in G.edges(data=True):
    print(f"Edge {edge[0]}-{edge[1]}: {edge[2]}")
# %%
G.nodes[0]
# %%
adj_matrix = nx.adjacency_matrix(G).todense()

print("Adjacency Matrix:")
print(adj_matrix)
# %%
G.adj[0]
# %%
import matplotlib.pyplot as plt
import networkx as nx


pos = nx.spring_layout(G, seed=42) 

plt.figure(figsize=(12, 10)) 
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=100, 
    node_color="skyblue", 
    edge_color="gray",  
    font_size=10,  
    font_color="black",  
    font_weight="bold", 
    width=1.0, 
    alpha=0.7  
)

plt.title("Real Estate Network Graph", fontsize=15, fontweight='bold')
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)

# نمایش پلات
plt.show()

# %%
# Save graph to GraphML file
nx.write_gml(G, "real_estate_network.gml")
# %%
# import networkx as nx
# import matplotlib.pyplot as plt

# # Read the graph from a GML file
# G = nx.read_gml("real_estate_network.gml")

# # Print basic information about the graph
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())

# # Print some nodes and their attributes
# print("\nNodes and their attributes:")
# for node in list(G.nodes(data=True))[:5]:  # Show first 5 nodes
#     print(node)

# # Print some edges and their attributes
# print("\nEdges and their attributes:")
# for edge in list(G.edges(data=True))[:5]:  # Show first 5 edges
#     print(edge)

# # Draw the graph
# pos = nx.spring_layout(G, seed=42)  # Use a layout algorithm
# plt.figure(figsize=(12, 10))
# nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", edge_color="gray", font_size=8, width=0.5)
# plt.title("Loaded Real Estate Network")
# plt.show()

# %%
import networkx as nx
import pandas as pd

# Read the graph from the GML file
G = nx.read_gml("real_estate_network.gml")

# Convert nodes to DataFrame
nodes_data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
nodes_data.reset_index(inplace=True)
nodes_data.rename(columns={'index': 'node_id'}, inplace=True)

# Convert edges to DataFrame
edges_data = pd.DataFrame(list(G.edges(data=True)), columns=['source', 'target', 'attributes'])

# Expand attributes dictionary into separate columns
edges_data = pd.concat([edges_data.drop(['attributes'], axis=1), edges_data['attributes'].apply(pd.Series)], axis=1)


edges_data
# %%
import networkx as nx
import pandas as pd

# Read the graph from the GML file
G = nx.read_gml("real_estate_network.gml")

# Convert nodes to DataFrame
nodes_data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
nodes_data.reset_index(inplace=True)
nodes_data.rename(columns={'index': 'node_id'}, inplace=True)

# Convert edges to DataFrame
edges_data = pd.DataFrame(list(G.edges(data=True)), columns=['source', 'target', 'attributes'])
edges_data = pd.concat([edges_data.drop(['attributes'], axis=1), edges_data['attributes'].apply(pd.Series)], axis=1)

# # Print columns before merging
# print("Columns before merging:")
# print(edges_data.columns)
# print(nodes_data.columns)

# Merge node attributes into edges DataFrame
edges_data = edges_data \
    .merge(nodes_data, left_on='source', right_on='node_id', suffixes=('', '_source')) \
    .merge(nodes_data, left_on='target', right_on='node_id', suffixes=('', '_target'))

# Print columns after merging
print("\nColumns after merging:")
print(edges_data.columns)

# Drop redundant columns
edges_data = edges_data.drop(columns=['node_id_source', 'node_id_target'], errors='ignore')

# Print the merged DataFrame for inspection
print("\nMerged DataFrame:")
edges_data

# %%
nodes_data
# %%
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from geopy.distance import geodesic
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# Assuming df_shuffled and G have been created previously

# Create a GeoDataFrame for use with contextily
gdf = gpd.GeoDataFrame(df_shuffled, 
                       geometry=gpd.points_from_xy(df_shuffled.longitude, df_shuffled.latitude),
                       crs="EPSG:4326")

# Convert CRS to a metric coordinate system (e.g., "EPSG:3857")
gdf = gdf.to_crs(epsg=3857)

# Create the plot and axes
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the base map
gdf.plot(ax=ax, color='red', markersize=10, alpha=0.7, edgecolor='k')

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')

# Convert geographic coordinates to metric coordinates
pos_metric = {k: (v[1], v[0]) for k, v in pos.items()}  # Switch latitude and longitude

# Draw the graph with enhanced settings
nx.draw(G, pos=pos_metric, ax=ax, with_labels=True, node_size=100, node_color="skyblue", edge_color="gray", font_size=8, width=0.5, alpha=0.7)

# Add the OpenStreetMap basemap with a suitable zoom level
ctx.add_basemap(ax, crs=gdf.crs.to_string(), zoom=10, alpha=0.3)

# Display the plot
plt.title("Real Estate Locations and Their Connections")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
# %%
# !pip install geopandas -U
# !pip show geopandas
# %%
!pip install contextily
# %%
!pip install shapely
# %% [markdown]
# ### BOW
# %% [markdown]
# log transform
# %%
df['price']=np.log(df['price'])
df['price'].skew()
df['area']=np.log(df['area'])
df['area'].skew()
# %%
df['final_description'].fillna("",inplace=True)
df_test['final_description'].fillna("",inplace=True)
# %%

# %%
from sklearn.feature_extraction.text import CountVectorizer
# %%
bow=CountVectorizer(min_df=0.01)
x_docTerm=bow.fit_transform(df['final_description'])
# %%
x_docTerm.shape
# %% [markdown]
# ### Tf-idf
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
# %%
tf=TfidfVectorizer(min_df=0.01)
x_docTerm_tf=tf.fit_transform(df['final_description'])
# %% [markdown]
# ## Model training
# %%
def fit_and_evaluate(model, X_train, y_train, X_test, y_test, tolerance=0.2):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # R2 Score
    r2 = r2_score(y_test, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape= mean_absolute_percentage_error(y_test, y_pred)
    
    #range accuracy
    correct_count = 0
    total_count = len(y_test)

    for true, predicted in zip(y_test, y_pred):
        if abs(true - predicted) <= tolerance*true:
            correct_count += 1

    accuracy = (correct_count / total_count)
    
    #Scatterplot
    plt.scatter(y_test, y_pred)
    plt.show()
    
    return r2, mape , accuracy
# %% [markdown]
# ### feature selection
# %%
region_bin=pd.cut(df['region'] ,6)
price_bin=pd.cut(df['price'] , 2)
area_bin=pd.cut(df['area'] , 2)
type_bin=pd.cut(df['type'] , 2)

df['price_bin']=price_bin
df['region_bin']=region_bin
df['type_bin']=type_bin
df['area_bin']=area_bin
X = df.drop(['price' , 'description' , 'neighborhood' , 'price_bin' , 'region_bin' , 'type_bin' , 'area_bin','description_cleaned','final_description'], axis=1).values
y = df['price'].values
# %%
X_bow=np.hstack((X,x_docTerm.toarray()))
# %%
X_tf=np.hstack((X,x_docTerm_tf.toarray()))
# %%
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.05, stratify=df[['region_bin' , 'price_bin' , 'type_bin' , 'area_bin']] ,random_state=42)
# %% [markdown]
# ## Basic model
# %%
model = LinearRegression()

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# ## DecisionTree
# %%
model = DecisionTreeRegressor(criterion='absolute_error',  max_depth=10 , random_state=5 )

r2, mape , accuracy = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# #### Model 1
# %%
model_1 = GradientBoostingRegressor(loss='huber' , n_estimators=250 ,  max_depth=10 , random_state=5  )

r2, mape , accuracy = fit_and_evaluate(model_1,X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# ### hist
# %%
hist_1 = HistGradientBoostingRegressor ( loss='absolute_error' , learning_rate=0.5 , max_depth=150 , random_state=5 , max_leaf_nodes=100 
                                        , min_samples_leaf=5)

r2, mape , accuracy = fit_and_evaluate(hist_1 ,  X_train , y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# #### Model 2
# %%
model_2 = GradientBoostingRegressor(loss='squared_error' , n_estimators=250 ,  max_depth=10 , random_state=5 )

r2, mape , accuracy = fit_and_evaluate(model_2,X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# ### Model 3
# %%
from sklearn.svm import SVR

# %%
model_3 = SVR(kernel='linear')

r2, mape , accuracy = fit_and_evaluate(model_3,X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# ### Model 4
# %%
from xgboost import XGBRegressor

# %%
model_4 = XGBRegressor(n_estimators=500 , learning_rate=0.01 , gamma=0, subsample=0.75 ,max_depth=10,random_state=5,
                   min_child_weight=1,colsample_bytree=0.8)

r2, mape , accuracy = fit_and_evaluate(model_4, X_train, y_train, X_test, y_test)

print("R2 Score:", r2)
print("MAPE:", (1-mape))
print("acc:", accuracy)
print('final score: ' ,  (r2+(1-mape)+accuracy)/3)
# %% [markdown]
# ## Prediction
# %%
def config_test_data(data):
    
    #fillna
    data=data.fillna('')
    
    #extract 
    data['region']= data['neighborhood'].apply(lambda x: (str(x).strip().split(" ")[1])  if (len(str(x).strip().split(" ")))>1 else np.nan).astype('float32')
    
    # 1 : vilaea & 0 : apartemani
    data['type']=data['type'].apply(lambda x: 0 if x=='فروش آپارتمان' else 1)
    

    
    #extract sanad
    data['sanad']=0
    data['sanad'].iloc[data[data.description.str.contains('سند')].index]=1
    data['sanad'].iloc[data[data.description.str.contains('سند آماده')].index]=1
    data['sanad'].iloc[data[data.description.str.contains('تک‌برگ')].index]=1
    data['sanad'].iloc[data[data.description.str.contains('سند تک')].index]=1
    data['sanad'].iloc[data[data.description.str.contains("قول نامه ای")].index]=2
    data['sanad'].iloc[data[data.description.str.contains("قول نامه")].index]=2
    data['sanad'].iloc[data[data.description.str.contains("قولنامه")].index]=2
    data['sanad'].iloc[data[data.description.str.contains("قولنامه‌ای")].index]=2
    data['sanad'].iloc[data[data.description.str.contains("قول‌نامه‌ای")].index]=2
    data['sanad'].iloc[data[data.description.str.contains('سند تک‌برگ')].index]=1
    data['sanad'].iloc[data[data.description.str.contains('سند‌تک‌برگ')].index]=1
    data['sanad'].iloc[data[data.description.str.contains('سند : سایر')].index]=3
    
    data['area']=np.log(data['area'])    #neww

    data=data.drop(['description' , 'neighborhood',"description_cleaned"] , axis=1)

    return data    
# %%
df_test=config_test_data(df_test)
# %%
df_test
# %%
x_docTerm_test=bow.transform(df_test['final_description'])
# %%
x_test=df_test.drop(columns=["final_description","price"]).values
# %%
x_test=np.hstack((x_test,x_docTerm_test.toarray()))
# %% [markdown]
# #### Prediction 1
# %%
y_pred = model_1.predict(x_test)
# %%
data_predict={"price" : np.exp(y_pred)}
# %%
a=pd.DataFrame(data_predict)
a
# %%

# %% [markdown]
# # 100% train
# %%
model_1 = GradientBoostingRegressor(loss='huber' , n_estimators=250 ,  max_depth=10 , random_state=5  )


# Fit the model
model_1.fit(X_bow , y)

# %%
