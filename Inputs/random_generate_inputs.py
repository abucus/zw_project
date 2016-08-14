from __future__ import print_function
import random
from random import randint as uniform_number
from random import gauss as normal_number
import csv
import networkx as nx
import os
import xlrd


def generate_input(path='./generated/', output_path='./case1/', project_num_range=[40, 45],
                   resource_type_num_range=[50, 150],
                   non_renew_resource_type_num_range=[50, 140], project_review_range=[1, 5],
                   project_review_range2=[300, 350], activity_num_range=[30, 45], project_activity_duration=[10, 30],
                   resource_project_demand=[10, 100], supplier_num_for_resource=[40, 45],
                   resource_supplier_capacity=[500, 80], resource_supplier_release_time=[20, 5],
                   supplier_project_cost=[1000, 100], supplier_project_shipping_time=[1, 7]):
    # generate activity network pool
    random.seed(11)  # specify random set
    dir = os.listdir(path)
    G_dir = {}
    for row in dir:
        # print row
        graph_name = row
        node_number = int(row.split('_')[1])
        file = path + '\\' + row
        reader = csv.reader(open(file, 'r'))
        this_graph = nx.DiGraph()
        if node_number in G_dir:
            for row1 in reader:
                if len(row1) > 1:
                    # print str(row1[1]).split(' '),'total',len(row1[1])
                    for row2 in str(row1[1]).split(' '):
                        if len(row2) > 0:
                            # try:
                            # print str(row2),'yes'
                            this_graph.add_edge(int(row1[0]), int(row2))
                            # except:
                            #     pass
            G_dir[node_number].append(this_graph)
        else:
            G_dir[node_number] = []
            for row1 in reader:
                if len(row1) > 1:
                    for row2 in str(row1[1]).split(' '):
                        if str(row2) != "''":
                            print(str(row2))
                            this_graph.add_edge(row1[0], row2)
            # for row22 in this_graph.edges():
            #     print row22
            G_dir[node_number].append(this_graph)

    project_n = uniform_number(*project_num_range)  # project number
    project_list = ["P" + str(i) for i in range(1, project_n + 1)]
    # print project_list

    resource_type_n = uniform_number(*resource_type_num_range)  # NK type number
    resource_type_list = ["NK0g" + str(i) for i in range(1, resource_type_n + 1)]

    non_renew_resource_type_n = uniform_number(*non_renew_resource_type_num_range)  # RK type number
    non_renew_resource_type_list = ["RK0" + str(i) for i in range(1, non_renew_resource_type_n + 1)]

    # generate project_review_due
    writer = csv.writer(open(output_path + "project_review_due.csv", 'w', newline=''))
    for row in project_list:
        writer.writerow([row, uniform_number(*project_review_range),
                         uniform_number(*project_review_range2)])  # modify project_revie_due

    # generate project activity_duration,project_activity
    writer2 = csv.writer(open(output_path + "project_activity_duration.csv", 'w', newline=''))
    writer1 = csv.writer(open(output_path + "project_activity.csv", 'w', newline=''))

    for row in project_list:
        i = 0
        this_activity_num = uniform_number(*activity_num_range)  # activity number of each project
        # this_activity_num=this_activity_num-this_activity_num%5
        # print this_activity_num
        print(G_dir.keys())
        G_dir[this_activity_num]
        acitivity_graph = G_dir[this_activity_num][uniform_number(0, len(G_dir[this_activity_num]) - 1)]
        for k in range(this_activity_num):
            i += 1
            # print len(resource_type_list),resource_type_n

            # NK activity required resource number and list
            resource_require_list = random.sample(resource_type_list, uniform_number(5, resource_type_n))

            # RK activity required resource number ad list
            non_renew_resource_require_list = random.sample(non_renew_resource_type_list,
                                                            uniform_number(5, non_renew_resource_type_n))
            this_resource_list = ''
            for row22 in resource_require_list:
                this_resource_list += row22 + str(" ")
            this_non_resource_list = ''
            for row22 in non_renew_resource_require_list:
                this_non_resource_list += row22 + str(" ")
            if i < 10:
                node1 = "T00" + str(i)
            elif i < 100:
                node1 = "T0" + str(i)
            else:
                node1 = "T" + str(i)
            my_edges = []
            for kk in range(this_activity_num):
                if kk < 10:
                    node2 = "T00" + str(kk)
                elif i < 100:
                    node2 = "T0" + str(kk)
                else:
                    node2 = "T" + str(kk)

                if node1 != node2 and (i, kk) in acitivity_graph.edges():
                    # print 'yes'
                    that_node = node2
                    my_edges.append(node2)
            if len(my_edges) > 0:
                print(my_edges)
                writer1.writerow(
                    [row, node1, this_resource_list, this_non_resource_list, str(my_edges)[1:-1].replace(',', ' ')])
            else:
                writer1.writerow([row, node1, this_resource_list, this_non_resource_list])
            writer2.writerow([row, node1, uniform_number(*project_activity_duration)])  # project activity duration

    # generate resource_project
    writer3 = csv.writer(open(output_path + "resource_project.csv", 'w', newline=''))
    a = []
    a.append('Material')
    for row in project_list:
        a.append(row)
    writer3.writerow([row for row in a])
    for row in resource_type_list:
        a = []
        a.append(row)
        for row in project_list:
            a.append(uniform_number(*resource_project_demand))  # prject required resource demand
        writer3.writerow([rows for rows in a])

    # generate resource_supplier
    writer4 = csv.writer(open(output_path + 'resource_supplier.csv', 'w', newline=''))
    writer5 = csv.writer(open(output_path + 'supplier_project_cost.csv', 'w', newline=''))
    writer6 = csv.writer(open(output_path + 'supplier_project_shipping.csv', 'w', newline=''))
    current_supplier = 0
    for row in resource_type_list:
        this_resource_supplier_num = uniform_number(
            *supplier_num_for_resource)  # how many supplier for this resource #####change running time
        for row2 in range(current_supplier, current_supplier + this_resource_supplier_num):
            writer4.writerow([row, "S" + str(row2), int(normal_number(*resource_supplier_capacity)),
                              int(normal_number(
                                  *resource_supplier_release_time))])  # resource- supplier capacity & release time
            arr1 = []
            arr2 = []
            arr1.append(row)
            arr1.append("S" + str(row2))
            arr2.append(row)
            arr2.append("S" + str(row2))
            for row3 in project_list:
                arr1.append(normal_number(*supplier_project_cost))  # supplier to project cost
                arr2.append(uniform_number(*supplier_project_shipping_time))  # supplier to project shipping time
            writer5.writerow([rowss for rowss in arr1])
            writer6.writerow([rowss for rowss in arr2])
        current_supplier += this_resource_supplier_num


# generate supplier_project_cost,supplier_project_shipping

#####project weight w

def solvable_check(path='./case1/'):
    from input_data import load_data
    D = load_data(path)
    resources = list(set([r for (r, p) in D.resource_project_demand]))
    resources.sort()

    suppliers = list(set([s for (r, s) in D.resource_supplier_capacity]))
    suppliers.sort()
    # print(resources)

    for r in resources:
        # demands of resource r for different projects order by demand from high to low
        demands = []
        for p in D.project_list:
            if D.resource_project_demand[r, p] > 0:
                demands.append((p, D.resource_project_demand[r, p]))

        demands.sort(key=lambda x: x[1], reverse=True)
        # print('-' * 50)
        # print('allocate resource %s' % r)
        # print('demands:\n%r' % demands)
        # supply of resource r from different suppliers
        supply = [(s, D.resource_supplier_capacity[r_, s]) for (r_, s) in D.resource_supplier_capacity if
                  r_ == r]
        # print('supply:\n%r' % supply)

        for (p, demand) in demands:
            # print('trying to find %s for %s (%r)' % (r, p, demand))
            # print(len(supply), 'all:%r' % supply)
            supplier_candidates = [s for (s, capacity) in supply if
                                   capacity >= demand]

            if not supplier_candidates:
                print('Unsolvable')
                return False
        else:
            return True


if __name__ == '__main__':
    import os

    print(os.getcwd())
    generate_input(project_num_range=[15, 20],
                   resource_type_num_range=[10, 12],
                   non_renew_resource_type_num_range=[9, 12],
                   supplier_num_for_resource=[8, 10])
    print(solvable_check())
