import networkx as nx
import matplotlib.pyplot as plt
import csv
import random

G=[]
for network_num in range(5,100): #activity number size from 5 to 20
    for seedss in range(100):
        G1=nx.fast_gnp_random_graph(network_num,0.1,seed=seedss,directed=True)
        DAG=nx.DiGraph([(u,v,{'weight':random.randint(-10,10)}) for (u,v) in G1.edges() if u<v])
        print(nx.is_directed_acyclic_graph(DAG))
        if DAG not in G and nx.is_directed_acyclic_graph(DAG):
            G.append(DAG)
            file=open('generated//'+'size_'+str(network_num)+'_'+str(seedss)+'.csv','wb')
            writer=csv.writer(file)
            for row in DAG.nodes():
                this_edges=[]
                for row1 in DAG.nodes():
                    if (row,row1) in DAG.edges():
                        this_edges.append(row1)
                if len(this_edges)>0:
                    writer.writerow([row,str(this_edges)[1:-1].replace(',',' ')])
                else:
                    writer.writerow([row])
            file.close()
            nx.draw(DAG)
            plt.savefig('generated_image//'+'size_'+str(network_num)+'_'+str(seedss)+'.png')
            plt.clf()
plt.show()