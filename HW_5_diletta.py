#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import collections
import pandas as pd 
import numpy as np
from io import StringIO 
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from shapely.geometry import Point, Polygon
import json, string
import requests
import geocoder
from folium import Map, Marker, GeoJson, LayerControl
from ediblepickle import checkpoint
from tqdm import tqdm_notebook
import os 
import folium
from folium import plugins
from collections import defaultdict


# In[ ]:


with open('coordinates.co','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Node_data = pd.DataFrame([map(int,raw.strip("\n").strip('v ').split()) for raw in out],columns = ['Id_Node', 'Latitude', 'Longitude'])

#info_norm =(df[['Longitude','Latitude']] - df[['Longitude','Latitude']].mean())/df[['Longitude','Latitude']].std()
#sns.scatterplot(x = 'Longitude', y = 'Latitude'  , data = info_norm)

with open('distancegraph.gr','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Distance_data = pd.DataFrame([map(int,i.strip('\n').strip("a ").split()) for i in out],columns =['Id_Node1', 'Id_Node2', 'd(Id_Node1,Id_Node2)'])

with open('traveltimegraph.gr','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Travel_time_data = pd.DataFrame([map(int,i.strip('\n').strip("a ").split()) for i in out],columns =['Node1', 'Node2', 't(Id_Node1, Id_Node2)'])


# In[ ]:


coordinates=Node_data
distance=Distance_data
traveltime=Travel_time_data

del Node_data
del Distance_data
del Travel_time_data

distance.rename({'Id_Node1': 'Node1', 'Id_Node2': 'Node2', 'd(Id_Node1,Id_Node2)':'Distance'}, axis=1,inplace=True)
traveltime.rename({'t(Id_Node1, Id_Node2)': 'TimeTravel'}, axis=1,inplace=True)

coordinates['Latitude']=coordinates['Latitude']/10**6
coordinates['Longitude']=coordinates['Longitude']/10**6


# In[ ]:


def dictionary(distance):
    d = defaultdict(list)
    for index in tqdm_notebook(distance.index, desc='Fill the Dictionary'):
        d[distance["Node1"].iloc[index]].append((distance["Node2"].iloc[index],distance["Distance"].iloc[index],traveltime["TimeTravel"].iloc[index],1))
    np.save('Graph_dict.npy', d)


# In[ ]:

dictionary(distance)
d = np.load('Graph_dict.npy',allow_pickle='TRUE').item() # load the nested dict


# In[ ]:


def spanning_tree_distance():
    print('Choose from which node start')
    node=int(input())
    mst = defaultdict(set)
    visited = set([node])
    edges = [(dist,node,to) for to,dist,_,_ in d[node]]
    heapq.heapify(edges)

    while edges:
        dist, frm, to = heapq.heappop(edges) #Pop and return the smallest item from the heap, maintaining the heap invariant.
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next, dist,_,_ in d[to]:
                if to_next not in visited:
                    heapq.heappush(edges, (dist, to, to_next)) #Push the value item into the heap, maintaining the heap invariant.
    return mst


# In[ ]:


def spanning_tree_time():
    print('Choose from which node start')
    node=int(input())   
    mst = defaultdict(set)
    visited = set([node])
    edges = [(time,node,to) for to,_,time,_ in d[node]]
    heapq.heapify(edges)

    while edges:
        time, frm, to = heapq.heappop(edges) #Pop and return the smallest item from the heap, maintaining the heap invariant.
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next,_,time,_ in d[to]:
                if to_next not in visited:
                    heapq.heappush(edges, (time, to, to_next)) #Push the value item into the heap, maintaining the heap invariant.
    return mst


# In[ ]:


def spanning_tree_weight():
    print('Choose from which node start')
    node=int(input())
    mst = defaultdict(set)
    visited = set([node])
    edges = [(weight,node,to) for to,_,_,weight in d[node]]
    heapq.heapify(edges)

    while edges:
        weight, frm, to = heapq.heappop(edges) #Pop and return the smallest item from the heap, maintaining the heap invariant.
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next,_,time,_ in d[to]:
                if to_next not in visited:
                    heapq.heappush(edges, (weight, to, to_next)) #Push the value item into the heap, maintaining the heap invariant.
    return mst


# In[ ]:


from collections import defaultdict
nodini=[]
print('Choose how many nodes consider for the spanning tree')
n=int(input())
print('Choose which nodes consider')
for i in range(0,n):
    nodini.append(int(input()))
print('Choose which method use for find the smartest Network:Time, Weight, Distance. p.s. If you are not interest just write "quit"') 

a=input().lower()
if a=='time':
    b=dict(spanning_tree_time())


elif a=='weight':
    b=dict(spanning_tree_weight())





elif a=='distance':
    b=dict(spanning_tree_distance())



elif a=='quit':
    print(':(')

else:
    print('What you said? please repeat!')




# In[ ]:


#frontend

Values=[]
for i in range(len(nodini)):

    try:
        Values.append(b[nodini[i]])
    except:
          continue

result_spanning = pd.DataFrame(list(b.items()), columns=['Id_Node', 'Connection'])

lista = []
for i in range(len(Values)):
    lista.append(result_spanning[result_spanning['Connection']==Values[i]])

result_spanning_1 = pd.concat(lista)

df_final = pd.merge(result_spanning_1, coordinates, on='Id_Node', how='inner')
df_final_1 = pd.merge(result_spanning, coordinates, on='Id_Node', how='inner')


# In[ ]:


def visualization_spanning_tree():    
#built the map centralized on the median of the coordinates  
    map = folium.Map(location=[np.median((df_final['Longitude']).tolist()),
                               np.median((df_final['Latitude']).tolist())], default_zoom_start=15)
    
#add marker, set of nodes result of spanning tree algorithm
    for i in tqdm_notebook(range(0,len(df_final))):
        folium.CircleMarker(location = [((df_final['Longitude']).values)[i], 
                                                         ((df_final['Latitude']).values)[i]],
                                                 number_of_sides = 4,
                                                 radius = 10,
                                                 weight = 8,
                                                 color='red',
                                                 fill_opacity = 0.8).add_to(map)
    for i in tqdm_notebook(range(0,len(df_final))):
        folium.Marker(location = [((df_final['Longitude']).values)[i], 
                                                         ((df_final['Latitude']).values)[i]],
                                                 number_of_sides = 4,
                                                 radius = 10,
                                                 weight = 8,
                                                 fill_opacity = 0.8).add_to(map)

#adding connection between two node
    for i in tqdm_notebook(range(0,len(df_final)-1)):
        folium.PolyLine(locations = [(((df_final['Longitude']).values)[i], 
                                      ((df_final['Latitude']).values)[i]), 
                                     (((df_final['Longitude']).values)[i+1], 
                                      ((df_final['Latitude']).values)[i+1])], 
                        line_opacity = 0.5,color='red').add_to(map)
    folium.CircleMarker([((df_final['Longitude']).values)[i],((df_final['Latitude']/1000000).values)[i]],
                        radius=15, color='red').add_to(map)
#weight represent the dimension of the node
#radius the shape
#opacity 
    map.save("Spanning_tree.html")

    return map


# In[ ]:


visualization_spanning_tree()



