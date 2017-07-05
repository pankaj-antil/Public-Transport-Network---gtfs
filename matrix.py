import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


#-----------------------------------------------------------------------------------------------------------------------
              # CONSTRUCTING ADJACENCY MATRIX

df = pd.read_csv('stops.csv')
stopid_col = df.stop_id  
length = len(stopid_col)


key = []
for i in range(0,length) :
    key.append(i)
 
col_names = ['stop_id','stop_code','stop_name','stop_desc','stop_lat','stop_lon','zone_id','stop_url','location_type','parent_station'] 
data = pd.read_csv('stops.csv',names=col_names)
list1 = data.stop_id.tolist()
del list1[0]


#indexing stopids
d = {}
d = dict(zip(list1, key))

    


mat= np.zeros(shape=(length,length))

colnames = ['trip_id','arrival_time','departure_time','stop_id','stop_sequence','stop_headsign','pickup_type',
            'drop_off_type','shape_dist_traveled','timepoint','continuous_drop_off','continuous_pickup']
data = pd.read_csv('stop_times.csv', names=colnames)
list2 = data.stop_id.tolist()
del list2[0]

list3 = data.trip_id.tolist()
del list3[0]

l=len(list2)
total_weight=0

for i in range(1,l) :
    if(list3[i]==list3[i-1]) :
	    total_weight = total_weight + 1
                         

for i in range(1,l) :
    if(list3[i]==list3[i-1]) :
	    mat[d.get(list2[i-1])][d.get(list2[i])] += 1/total_weight
		
#printing matrix in a file
names = [_ for _ in list1]
df = pd.DataFrame(mat, index=names, columns=names)
df.to_csv('df.csv', index=True, header=True, sep=' ')	
#matrix output 	
print (mat,'\n')

f= open('edgelist.txt','w+')
for i in range(1,l) :
    if(list3[i]==list3[i-1]) :
	    f.write('%d %d\n'%(int(list2[i-1]),int(list2[i])))
        

#----------------------------------------------------------------------------------------------------------
            # DRAWING GRAPH 
G = nx.Graph()
f = open('edgelist.txt', 'r')

i=0
for line in f.readlines():
    tmp = line.strip()
    tmp = tmp.split(" ")
    from_edge = int(tmp[0])
    to_edge = int(tmp[1])
   
    # ensure the column has at least one value before printing
    if (from_edge, to_edge) in G.edges():
        G[from_edge][to_edge]['weight'] += 1/total_weight
    else:
        # new edge. add with weight=1
        G.add_edge(from_edge, to_edge, weight=1/total_weight)
        

# writing weighted edgelist from graph in a file
nx.write_weighted_edgelist(G, 'test.weighted.edgelist')


elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.03]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.03]


# positions for all nodes
pos=nx.spring_layout(G) 
#labels
nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')
# nodes
nx.draw_networkx_nodes(G,pos,node_size=500)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,width=2)
nx.draw_networkx_edges(G,pos,edgelist=esmall,width=2,alpha=0.5,edge_color='b',style='dashed')
                    
                   
plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display


#-------------------------------------------------------------------------------------------------------
           # PLOTTING DEGREE DISTRIBUTION FOR GRAPH

degs = {}
for n in G.nodes() :
    deg = G.degree(n)
    if deg not in degs :
       degs[deg] = 0
    degs[deg] +=1
items = sorted(degs.items())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([k for (k,v) in items],[v for (k,v) in items])
ax.set_xscale('log')
ax.set_yscale('log')
plt.title('Public Transport Network Degree Distribution')
fig.savefig('degree_distribution.png')
plt.show()



#-------------------------------------------------------------------------------------------------------------
             #  CONSTRUCTING SUPER ADJACENCNY MATRIX 
# (x,y) coordinates
x = data.stop_lat.tolist()
del x[0]
y = data.stop_lon.tolist()
del y[0]

d2 = {}
d2 = dict(zip(x,y))





for i in range (0,length) :
    pos[from_edge]=(x[d[from_edge]],y[d[from_edge]])
 pos[to_edge]= (x[d[to_edge]],y[d[to_edge]])








    
 