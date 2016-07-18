import random
import csv
import networkx as nx
import os
import xlrd
from __future__ import print_function

def uniform_number(a,b):
    return random.randint(a,b)

def normal_number(a,b):
    return random.gauss(a,b)

#generate activity network pool
random.seed(11) #specify random set
path1='C:\\Users\\Zhengwei\\Desktop\\zhengwei proposal\\final_project\Inputs\\generated'
dir=os.listdir(path1)
G_dir={}
for row in dir:
    # print row
    graph_name=row
    node_number=int(row.split('_')[1])
    file=path1+'\\'+row
    reader=csv.reader(open(file,'rb'))
    this_graph=nx.DiGraph()
    if node_number in G_dir:
        for row1 in reader:
            if len(row1)>1:
                # print str(row1[1]).split(' '),'total',len(row1[1])
                for row2 in str(row1[1]).split(' '):
                    if len(row2)>0:
                        # try:
                        # print str(row2),'yes'
                        this_graph.add_edge(int(row1[0]),int(row2))
                        # except:
                        #     pass
        G_dir[node_number].append(this_graph)
    else:
        G_dir[node_number]=[]
        for row1 in reader:
            if len(row1)>1:
                for row2 in str(row1[1]).split(' '):
                    if str(row2)!="''":
                        print(str(row2))
                        this_graph.add_edge(row1[0],row2)
        # for row22 in this_graph.edges():
        #     print row22
        G_dir[node_number].append(this_graph)




project_n=uniform_number(40,45) #project number
project_list=["P"+str(i) for i in range(1,project_n+1)]
# print project_list

resource_type_n=uniform_number(50,150) #NK type number
resource_type_list=["NK0g"+str(i) for i in range(1,resource_type_n+1)]

non_renew_resource_type_n=uniform_number(50,140) #RK type number
non_renew_resource_type_list=["RK0"+str(i) for i in range(1,non_renew_resource_type_n+1)]

#generate project_review_due
writer=csv.writer(open("case1\\project_review_due.csv",'wb'))
for row in project_list:
    writer.writerow([row,uniform_number(1,5),uniform_number(300,350)]) #modify project_revie_due

#generate project activity_duration,project_activity
writer2=csv.writer(open("case1\\project_activity_duration.csv",'wb'))
writer1=csv.writer(open("case1\\project_activity.csv",'wb'))

for row in project_list:
   i=0
   this_activity_num=uniform_number(30,45)#activity number of each project
   # this_activity_num=this_activity_num-this_activity_num%5
   # print this_activity_num
   acitivity_graph=G_dir[this_activity_num][uniform_number(0,len(G_dir[this_activity_num])-1)]
   for k in range(this_activity_num):
       i+=1
       # print len(resource_type_list),resource_type_n
       resource_require_list=random.sample(resource_type_list,uniform_number(10,resource_type_n)) #NK activity required resource number and list
       non_renew_resource_require_list=random.sample(non_renew_resource_type_list,uniform_number(10,non_renew_resource_type_n)) #RK activity required resource number ad list
       this_resource_list=''
       for row22 in resource_require_list:
           this_resource_list+=row22+str(" ")
       this_non_resource_list=''
       for row22 in non_renew_resource_require_list:
           this_non_resource_list+=row22+str(" ")
       if i<10:
           node1="T00"+str(i)
       elif i<100:
           node1="T0"+str(i)
       else:
           node1="T"+str(i)
       my_edges=[]
       for kk in range(this_activity_num):
           if kk<10:
               node2="T00"+str(kk)
           elif i<100:
               node2="T0"+str(kk)
           else:
               node2="T"+str(kk)

           if node1!=node2 and (i,kk) in acitivity_graph.edges():
               # print 'yes'
               that_node=node2
               my_edges.append(node2)
       if len(my_edges)>0:
           print(my_edges)
           writer1.writerow([row,node1,this_resource_list,this_non_resource_list,str(my_edges)[1:-1].replace(',',' ')])
       else:
           writer1.writerow([row,node1,this_resource_list,this_non_resource_list])
       writer2.writerow([row,node1,uniform_number(10,30)]) #project activity duration

#generate resource_project
writer3=csv.writer(open("case1\\resource_project.csv",'wb'))
a=[]
a.append('Material')
for row in project_list:
    a.append(row)
writer3.writerow([row for row in a])
for row in resource_type_list:
    a=[]
    a.append(row)
    for row in project_list:
        a.append(uniform_number(10,100)) #prject required resource demand
    writer3.writerow([rows for rows in a])

#generate resource_supplier
writer4=csv.writer(open('case1\\resource_supplier.csv','wb'))
writer5=csv.writer(open('case1\\supplier_project_cost.csv','wb'))
writer6=csv.writer(open('case1\\supplier_project_shipping.csv','wb'))
current_supplier=0
for row in resource_type_list:
    this_resource_supplier_num=uniform_number(40,45) #how many supplier for this resource #####change running time
    for row2 in range(current_supplier,current_supplier+this_resource_supplier_num):
        writer4.writerow([row,"S"+str(row2),int(normal_number(500,30)),int(normal_number(20,5))]) #resource- supplier capacity & release time
        arr1=[]
        arr2=[]
        arr1.append(row)
        arr1.append("S"+str(row2))
        arr2.append(row)
        arr2.append("S"+str(row2))
        for row3 in project_list:
            arr1.append(normal_number(1000,10)) #supplier to project cost
            arr2.append(uniform_number(1,7)) #supplier to project shipping time
        writer5.writerow([rowss for rowss in arr1])
        writer6.writerow([rowss for rowss in arr2])
    current_supplier+=this_resource_supplier_num

#generate supplier_project_cost,supplier_project_shipping

#####project weight w


