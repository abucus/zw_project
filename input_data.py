#from random_input import *

# from gurobipy import *
# from RandomInput import *
import networkx as nx
import numpy as np
import random
# import operator
import csv
import numpy as np
import os

#parameters

#######################################################
path='./Inputs/case1'
# path='C:\Users\liujm\Desktop\zhengwei proposal\\final_project\Inputs'
#####load resource-project demand
print(os.getcwd())
resource_project_file=csv.reader(open(path+'/resource_project.csv','r'))
first_row=next(resource_project_file)
# next(resource_project_file)
resource_list=[]
project_list=first_row[1:]
resource_project_demand={}
for row in resource_project_file:
    if row[0] not in resource_list:
        resource_list.append(row[0])
        for i in range(1,len(row)):
            resource_project_demand[row[0],project_list[i-1]]=float(row[i])

resource_n=len(resource_list)
project_n=len(project_list)

#####load resource-supplier capacity and release time
resource_supplier_file=csv.reader(open(path+'\\resource_supplier.csv','r'))
resource_supplier_list={}
for row in resource_list:
    resource_supplier_list[row]=[]
resource_supplier_capacity={}
resource_supplier_release_time={}
for row in resource_supplier_file:
    resource_supplier_capacity[row[0],row[1]]=float(row[2])
    resource_supplier_release_time[row[0],row[1]]=float(row[3])
    if row[0] in resource_supplier_list:
        resource_supplier_list[row[0]].append(row[1])

#####load supplier_project shipping time
supplier_project_shipping_file=csv.reader(open(path+'\\supplier_project_shipping.csv','r'))
supplier_project_shipping={}
for row in supplier_project_shipping_file:
    for j in range(project_n):
        supplier_project_shipping[row[0],row[1],project_list[j]]=float(row[j+2])

#####load supplier-project cost c[i,j,k]
supplier_project_cost_file=csv.reader(open(path+'\\supplier_project_cost.csv','r'))
c={}
for row in supplier_project_cost_file:
    for j in range(project_n):
        c[row[0],row[1],project_list[j]]=float(row[j+2])

#####load project activity
project_activity={}
for j in range(project_n):
    project_activity[project_list[j]]=nx.DiGraph()
project_activitis=csv.reader(open(path+'\\project_activity.csv','r'))
for row in project_activitis:
    this_project=row[0]
    resource_list=[]
    RK_resource_list=[]
    this_list=row[2].split(' ')
    RK_list=row[3].split(' ')
    for row1 in this_list:
        if 'N' in row1:
            resource_list.append(row1)
    for row2 in RK_list:
        if 'R' in row2:
            RK_resource_list.append(row2)

    project_activity[row[0]].add_node(row[1])
    project_activity[row[0]].node[row[1]]['resources']=resource_list
    project_activity[row[0]].node[row[1]]['rk_resources']=RK_resource_list
    # print project_activity[row[0]].node[row[1]]


project_activitis=csv.reader(open(path+'\\project_activity.csv','r'))
for row in project_activitis:
    if len(row)>4:
        for row1 in row[4]:
            # print row[1],str(row[4]).split("'")[1]
            project_activity[row[0]].add_edge(row[1],str(row[4]).split("'")[1])

project_activity_durations=csv.reader(open(path+'\\project_activity_duration.csv','r'))
for row in project_activity_durations:
    project_activity[row[0]].node[row[1]]['duration']=float(row[2])

#####load project review duration
review_duration=[]
project_review_due=csv.reader(open(path+'\\project_review_due.csv','r'))
for row in project_review_due:
    review_duration.append(float(row[1]))

w = [1 for i in range(project_n)]  # need to change, project budget?
DD = [random.randint(30, 35) for i in range(project_n)]  # project_due_day

# ##################################################
#
# # Constant
# B=500000000
B = 800000000
M = 100000000

# G1.add_edge()


