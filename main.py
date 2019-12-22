#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import queue as q
import numpy as np


# ## Preparation the data vefore starting to do the graph

# In[3]:


with open('/Users/macbook/Desktop/Lev/Sapienza/ADM_Labs/HW5/USA-road-d.CAL.co','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Node_data = pd.DataFrame([map(int,raw.strip("\n").strip('v ').split()) for raw in out],columns = ['Id_Node', 'Latitude', 'Longitude'])


# In[4]:


df = df[['Longitude','Latitude']]/10**6 # standart format


# In[5]:


with open('/Users/macbook/Desktop/Lev/Sapienza/ADM_Labs/HW5/USA-road-d.CAL.gr','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Distance_data = pd.DataFrame([map(int,i.strip('\n').strip("a ").split()) for i in out],columns =['Id_Node1', 'Id_Node2', 'd(Id_Node1,Id_Node2)'])


# In[6]:


with open('/Users/macbook/Desktop/Lev/Sapienza/ADM_Labs/HW5/USA-road-t.CAL.gr','r') as f:
    out = f.readlines()
out = out[7:]
# create DataFrame
Travel_time_data = pd.DataFrame([map(int,i.strip('\n').strip("a ").split()) for i in out],columns =['Node1', 'Node2', 't(Id_Node1, Id_Node2)'])


# ## Creating the graph for drawing

# In[ ]:


G = nx.Graph()
G.add_nodes_from(list(Distance_data['Id_Node1'].unique())) # add all unique nodes into the graph
for index, row in Distance_data.iterrows(): # add all eges with help of data frame which we created during the preparation data
    source = row[0]
    dest = row[1]
    G.add_edge(source, dest, weight = 1,attr_dict = row.to_dict())


#  Let's create out own data structure to represent the graph. We decided to do a defaultdict with list. the main idea is to have:
#  **{node: [(neibour_1, distance,time_dicstane),(neibour_2, distance,time_dicstane),...,(neibour_N, distance,time_dicstane)]}**

# In[ ]:


d = np.load('Graph_dict.npy',allow_pickle='TRUE').item() # load the nested dict


# ## Visualization

# In[3]:


def draw_map(df,set_nodes,path,type_of_visualization):
    # Make a data frame with dots to show on the map
    data = df.loc[path]
    points = []
    m = folium.Map(location=[data.mean().Longitude,data.mean().Latitude],width=750, height=500,zoom_start=15)
    # I can add marker one by one on the map
    for i in data.index:
        try:
            if i in set_nodes:
                folium.CircleMarker([float(data.loc[i]['Longitude']), float(data.loc[i]['Latitude'])],fill = True,fill_color = "grey",color = "grey").add_to(m)
            elif i in path and i not in set_nodes:
                folium.CircleMarker([data.loc[i]['Longitude'], data.loc[i]['Latitude']],fill = False).add_to(m)
            points.append(tuple([data.loc[i]['Longitude'], data.loc[i]['Latitude']]))
        except:
            pass
    # visualize the path or neighbourhood
    if type_of_visualization == 3:
        folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(m)
    elif type_of_visualization == 1 or type_of_visualization == 0:
        pass
    return m


# ## Functionality 0 - Find the Neighbours!
# It takes in input:
# 
# 1) A node v
# 
# 2) One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).
# 
# 3) A distance threshold d

# In[264]:


def neighbours(d,treshold,v,which_dist = 1):
    neighbours = []
    if which_dist == 1:
        type_d = 1
    elif which_dist == 2:
        type_d = 2
    lst_neibours = [{i[0]:i[type_d]} for i in d[v]]
    # take 1'st element because the dict always look like {neighbour : dist to him} 
    neighbours = [ (list(i.keys())[0],list(i.values())[0]) for i in lst_neibours if list(i.values())[0] <= treshold]
    return neighbours
            
def draw_neighbours(neighbours,v):
    G = nx.Graph()


    for node in neighbours:
        source = v
        dest = node[0]
        G.add_edge(source, dest, weight = node[1])

    pos = nx.spring_layout(G)  # positions for all nodes
    colors = range(len(neigh) + 1)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700,node_color='#A0CBE2')
    
    # edges
    nx.draw_networkx_edges(G, pos, edgelist = G.edges(),
                           width=3)


    # labels
    nx.draw_networkx_labels(G, pos)

    plt.axis('off')
    plt.show()         


# ## Functionality 1 - Find the Neighborhood! (because the task was confusing)
# It takes in input:
# 
# 1) A node v
# 
# 2) One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).
# 
# 3) A distance threshold d

# In[219]:


# find the nodes which are < d from the source node
def neighborhood(G,source,treshold,type_dist):
    v1 = source
    cost = 0
    toDoSet = [v1]
    costSet = {v1 : 0}
    doneSet = []
    
    while (len(toDoSet) != 0 ):
        v = toDoSet[-1] # take the last element
        # remove node from toDoset and append node to the doneSet 
        toDoSet.remove(v) 
        doneSet.append(v) 
        for node in [node for node in G[v]]: # go through the neighbours of the node
            costSet[node[0]] = costSet[v] + node[type_dist] # check if the dist is okey
            if node[0] not in toDoSet and node[0] not in doneSet and costSet[node[0]] < treshold:
                toDoSet = [node[0]] + toDoSet # add new elements at the begining because we always take the elements from the tail
        
    return doneSet 

def draw_neighborhood(G,a):
    g = G.subgraph(a)
    pos = nx.spring_layout(g)
    colors = range(len(a) + 1)
    nx.draw_networkx_nodes(g, pos, node_size=700,node_color='#A0CBE2')
    nx.draw_networkx_edges(g, pos, edgelist = g.edges(),
                               width=3)
    nx.draw_networkx_labels(g, pos)
    plt.show()
#a = neighborhood(d,1,2000,2)   
#draw_neighborhood(G,a)


# ## Functionality 3 - Shortest Ordered Route
# It takes in input:
# 
# 1) A node H
# 
# 2) A sequence of nodes p = [p_1, ..., p_n]
# 
# 3) The following distances function: network distance (all edges to have weight equal to 1).
# 
# Task: Implement an algorithm that returns the shortest walk that goes from H to p_n, and that visits in order the nodes in p.

# In[108]:


# Check if the nodes are connected
# the idia with toDoSet and doneSet 
def connected(G,source,dest):
    v1,v2 = source,dest
    toDoSet = [v1]
    doneSet = [v2]
    while (len(toDoSet) != 0 ):
        v = toDoSet[-1]
        toDoSet.remove(v)
        doneSet.append(v)
        for node in [node_ for node_,_,_,_ in G[v]]: # take the list of nodes ( neigh ) from [(node1,_,_,_),...] to [node1,...]
            if node == v2:
                return doneSet
            if node not in toDoSet and node not in doneSet:
                toDoSet = [node] + toDoSet
    return [] 
a = connected(d,source = 1,dest = 87)            


# In[138]:


# extract the path from dijkstra algo
def find_path(all_search, source, dest):
    path = [dest]
    v = all_search[dest]
    path.append(v)
    while v != source:
        v = all_search[v]  
        path.append(v)
    return list(reversed(path))

def dijkstra(G, source, dest, type_dist = 3):
    frontier = q.PriorityQueue() # need to use this structure because key = number of vertex priority = cost 
    frontier.put(source, 0) # as we always start from  vertex with the cost smallest it is very usefull 
    came_from = {}
    cost_so_far = {}
    came_from[source] = None # initial point
    cost_so_far[source] = 0 # cost for initial point
    
    while not frontier.empty():
        current = frontier.get() # take the smallest one
        
        if current == dest:
            break
        
        for neigh in G[current]: # look throw neigh
            new_cost = cost_so_far[current] + neigh[type_dist] # change the cost
            if neigh[0] not in cost_so_far or new_cost < cost_so_far[neigh[0]]:
                cost_so_far[neigh[0]] = new_cost
                priority = new_cost
                frontier.put(neigh[0], priority)
                came_from[neigh[0]] = current # dict of parent-neigh 
    
    return find_path(came_from, source, dest), cost_so_far[dest]
# Find the shortest path (with ordered nodes)
def sh_or_route(H, p, G, type_dist = 3):
    all_path = []
    all_cost = 0
    set_nodes = [H] + p
    if all([len(connected(G, set_nodes[i], set_nodes[i+1])) > 0 for i in range(len(set_nodes) - 1)]): # if nodes are connected
        for i in range(len(set_nodes) - 1):
            path, cost = dijkstra(G, set_nodes[i], set_nodes[i+1])
            all_cost += cost
            all_path.extend(path)
    else:
        print("Not possible")
    return all_path, all_cost
# this function draw the path (blue node is source, pint nodes is the set and red node is the destination)
def draw_path(g,path,p):
    color_map = []
    u_path = list(set(path))
    for i in range(len(u_path)):
        if i == 0:
            color_map.append('blue')
        elif i == len(u_path) - 1:
            color_map.append('red') 
        elif u_path[i] in p[:-1]:
            color_map.append('pink')  
        else: color_map.append('green')      
    nx.draw(g,node_color = color_map, )
    plt.title('Blue - source, pink - set of nodes, red - destination')

    plt.show()


# ## Question 2

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

coordinates = Node_data
distance = Distance_data
traveltime = Travel_time_data
distance.rename({'Id_Node1': 'Node1', 'Id_Node2': 'Node2', 'd(Id_Node1,Id_Node2)':'Distance'}, axis=1,inplace=True)
traveltime.rename({'t(Id_Node1, Id_Node2)': 'TimeTravel'}, axis=1,inplace=True)

# coordinates['Latitude']=coordinates['Latitude']/10**6
# coordinates['Longitude']=coordinates['Longitude']/10**6


# In[ ]:


def spanning_tree_time():
    print('Choose from which node start')
    node =int(input())   
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


def spanning_tree_weight():
    print('Choose from which node start')
    node=int(input())
    mst = defaultdict(set)
    visited = set([node])
    edges = [(weight,node,to) for to,_,_,weight in d[node]]
    heapq.heapify(edges)

    while edges:
        weight, frm, to = heapq.heappop(edges) # Pop and return the smallest item from the heap, maintaining the heap invariant.
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next,_,time,_ in d[to]:
                if to_next not in visited:
                    heapq.heappush(edges, (weight, to, to_next)) #Push the value item into the heap, maintaining the heap invariant.
    return mst


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
                                                 color ='red',
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


# ## Question 4

# In[ ]:


import pandas as pd
import networkx as nx
import itertools


tdist = pd.read_csv(r"CALtd.csv")
rdist = pd.read_csv(r"CALrd.csv")
nodexy = pd.read_csv(r"CALnodes.csv")

G1=nx.Graph()
G1.add_nodes_from(nodexy['id'])
edges1=rdist[['node1','node2']].values.tolist()
G1.add_edges_from(edges1)

G2=nx.Graph()
G2.add_nodes_from(nodexy['id'])
edges2=tdist[['node1','node2']].values.tolist()
G2.add_edges_from(edges2)

Gr = nx.dense_gnm_random_graph(20, 40, seed=None)


# In[ ]:


def ShortestCover(H,*p,distype):
#Made a list with all points (initial, mid and last)
    pts = []
    for i in p:
        pts.append(i)
    pts.insert(0,H)
    
#Given the distype, the graph made depends on either real distance or time distance.
    for i in p:
        pts.append(i)
    
    if distype == 1:
        G= G1
        
    elif distype == 2:
        G = G2
        
    elif distype ==3:
        G = Gr
    
    if nx.has_path(G,H,pts[-1]) == False:
            print("No possible path")
#Permutating the possible paths

    midpts = pts[1:-1]
    perm = list(itertools.permutations(midpts))
    
    out = [0] * (len(perm))
    
#Finding out path with shortest length

    for i in range(len(perm)):
        for j in range(len(perm[0])-1):
            out[i] += (nx.dijkstra_path_length(G,source = perm[i][j],target =perm[i][j+1])) 

    bpath = out.index(min(out))
    bmidi = list(perm[bpath])
    finalpath = [0]* len(pts)
    finalpath[0] = pts[0]
    finalpath[-1] = pts [-1]
    
    for i in range(1,len(pts)-1):
        finalpath[i] = bmidi [i-1]
        
#Given the order of points to be traversed calculated in the step above, find shortest paths between points, then add all paths together.
    allpath = []   
    for i in range(len(finalpath)-1):
        allpath.append(nx.dijkstra_path(G, source=finalpath[i], target = finalpath[i+1], weight='weight'))
    for lp in range(1,len(allpath)):
        allpath[lp].pop(0)
        
        
    allpath = list(itertools.chain(*allpath))
    
    return allpath

def drawSC(scpath):
    a = scpath
    
    mdf = rdist.loc[rdist['node1'].isin(a), ['color']] = 'yellow'
      
    g = nx.Graph()

    for i, elrow in rdist.iterrows():    
        g.add_edge(elrow[0], elrow[1], **elrow[2:].to_dict())

    for i, nlrow in nodexy.iterrows():
        nx.set_node_attributes(g, {nlrow['id']:  nlrow[1:].to_dict()})   

    node_positions = {node[0]: (node[1]['X'], -node[1]['Y']) for node in g.nodes(data=True)}
    edge_colors = [e[2]['color'] for e in list(g.edges(data=True))]


    nx.draw(g, pos=node_positions, edge_color=edge_colors, node_size=0.05, node_color='black', width = 2)
    plt.figure(figsize=(100, 60))
    plt.title('Shortest walk going through all nodes')
    plt.show()


# In[ ]:


print("Which Functionality do you want to use?")
while True:
    func = int(input())
    if func in [0,1,2,3,4]:
        break
    else:
        print("There is not your functionalyty, choose between 0 - 4")
if func == 0:
    # Organize the input and output of the programm
    while True:
        print("Which type of dist.?")
        type_of_dist = int(input())
        print("Choose the node v = ")
        v = int(input())
        print("Set the treshold = ")
        treshold = float(input())
        if v in list(d.keys()):
            print("Compiling the neighbour graph")
            neigh = neighbours(d,treshold,v,which_dist = type_of_dist)
            if len(neigh) == 0:
                print("There is not any neighbours with this treshold")
            else:  
                draw_neighbours(neigh,v)
                neibourhood = [v] + [i for i,_ in neigh]
                draw_map(df,neibourhood,neibourhood,type_of_visualization = 1).save("my_map0.html")
                print("Check the route into the your current filder")
            break
        else: 
            print("There is not node with the number ", v)
elif func ==1:
    # Organize the input and output of the programm
    while True:
        print("Choose the node v = ")
        v = int(input())
        print("Choose the type of distance: 1 - km, 2 - time, 3 - network dist")
        type_dist  = int(input())
        print("Set the treshold = ")
        treshold = float(input())
        if v in list(d.keys()):
            print("Compiling the neighbour graph ")
            a = neighborhood(d,v,treshold,type_dist)   
            if len(a) == 0:
                print("There is not any neighbours with this treshold")
            else:  
                draw_neighborhood(G,a)
                neibourhood = a
                draw_map(df,neibourhood,neibourhood,type_of_visualization = 1).save("my_map1.html")
                print("Check the route into the your current filder")
            break
        else: 
            print("There is not node number ", v)
elif func == 2:
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
        print('What did you say? please repeat!')
 
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

    df_final = pd.merge(result_spanning_1, coordinates, on = 'Id_Node', how = 'inner')
    df_final_1 = pd.merge(result_spanning, coordinates, on = 'Id_Node', how = 'inner')

elif func == 3:
    print("Input the source node:")
    H = int(input())
    print("Input the sequence of nodes, (for ex: '1 2 3 4 5 6 7')")
    p = list(map(int,input().split()))

    if all([len(d[i]) > 0 for i in [H] + p]):
        path,cost = sh_or_route(H,p,d,type_dist = 3)    
        g = G.subgraph(path)
        draw_path(g,path,p) 
        set_nodes = [H] + p
        draw_map(df,set_nodes,path,type_of_visualization = 3).save("my_map3.html")
        print("Check the route into the your current filder")
    else:
        print("Not possible, one of the nodes not in the graph")
elif func == 4:
    print("Which type of dist. you want? 1 or 2 or 3")
    type_of_dist = int(input())
    print("Insert main node")
    H = int(input())
    print("Insert set of nodes for ex : 1 2 3 4 5")
    p = list(map(int,input().split()))
    alpha = ShortestCover(H,p,type_of_dist)
    drawSC(alpha)


# In[ ]:





# In[ ]:




