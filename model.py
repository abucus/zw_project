from __future__ import print_function
from input_data import load_data
from gurobipy import *
from os import makedirs
from os.path import join
from os.path import exists
import time

# Create optimization model

def original_model(input_path, output_path):

    if not exists(output_path):
        makedirs(output_path)

    supplier_project_shipping, project_list, project_activity, DD, resource_supplier_capacity, \
    project_n, resource_project_demand, resource_supplier_list, M, c, B, resource_supplier_release_time, \
    review_duration, w = load_data(input_path)

    start_time = time.clock()
    m = Model('construction')
    ##############################################################
    m.params.presolve = 0
    # Create variables############################################
    #####supplier-project shipping decision x and shipping quality
    x = {}
    q = {}
    for (i, j, k) in supplier_project_shipping:
        # i resource, j supplier, k project
        x[i, j, k] = m.addVar(obj=0, vtype=GRB.BINARY, name="(x%s,%s,%s)" % (i, j, k))
        q[i, j, k] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(q%s,%s,%s)" % (i, j, k))
    #####Project complete data,Project Tadeness,construction completion time
    DT = {}
    TD = {}
    CT = {}
    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT-1)")  # project start time
    for j in range(project_n):
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT%d)" % j)  # project j complete time
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(TD%d)" % j)  # project j complete time
        CT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(CT%d)" % j)  # project j complete time

    #####Activity start time
    ST = []
    for j in range(project_n):
        ST.append({})
        for row in project_activity[project_list[j]].nodes():
            ST[j][row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(ST%d,%s)" % (j, row))

    #####Review sequence
    z = {}
    for i in range(project_n):
        for j in range(project_n):
            if i != j:
                z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (i, j))

    for j in range(project_n):
        z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (-1, j))

    #####
    y = {}
    for j in range(project_n):
        for row1 in project_activity[project_list[j]].nodes():
            for row2 in project_activity[project_list[j]].nodes():
                # print project_activity[project_list[j]].node[row1]
                if row1 != row2 and len(list(set(project_activity[project_list[j]].node[row1]['rk_resources']).intersection(
                        project_activity[project_list[j]].node[row2]['rk_resources']))) > 0:
                    y[j, row1, row2] = m.addVar(obj=0, vtype=GRB.BINARY, name="(y%d,%s,%s)" % (j, row1, row2))
    m.update()
    # create constrains#########################################
    #####Constrain 2: project complete data>due data
    for j in range(project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, DD[j], name="constraint_2_project_%d" % j)

    ##### constrain 3: supplier capacity limit
    for (row1, row2) in resource_supplier_capacity:
        m.addConstr(quicksum(q[row1, row2, project_list[j]] for j in range(project_n)), GRB.LESS_EQUAL,
                    resource_supplier_capacity[row1, row2], name="constraint_3_resource_%s_supplier_%s" % (row1, row2))

    #####constrain 4,6: project demand require; each project receive from one supplier for each resource
    for (row1, row2) in resource_project_demand:
        m.addConstr(quicksum(x[row1, i, row2] for i in resource_supplier_list[row1]), GRB.EQUAL, 1,
                    name="constraint_6_resource_%s_project_%s" % (row1, row2))
        m.addConstr(quicksum(q[row1, i, row2] for i in resource_supplier_list[row1]), GRB.GREATER_EQUAL,
                    resource_project_demand[row1, row2], name="constraint_4_resource_%s_project_%s" % (row1, row2))

    #####constrain 5: shipping constrain
    for (i, j, k) in q:
        # i resource, j supplier, k project
        m.addConstr(q[i, j, k], GRB.LESS_EQUAL, M * x[i, j, k],
                    name="constraint_5_resource_%s_supplier_%s_project_%s" % (i, j, k))

    #####constrain 7:budget limit
    expr = LinExpr()
    for (i, j, k) in q:
        expr = expr + c[i, j, k] * q[i, j, k]
    m.addConstr(expr, GRB.LESS_EQUAL, B, name="constraint_7")

    #####constrain 8: activity starting constrain
    for j in range(project_n):
        for row in project_activity[project_list[j]].nodes():
            for row1 in project_activity[project_list[j]].node[row]['resources']:
                m.addConstr(quicksum(x[row1, i, project_list[j]] * (
                    resource_supplier_release_time[row1, i] + supplier_project_shipping[row1, i, project_list[j]]) for i in
                                     resource_supplier_list[row1]), GRB.LESS_EQUAL, ST[j][row],
                            name="constraint_8_project_%d_activity_%s_resource_%s" % (j, row, row1))

    #####constrain 9 activity sequence constrain
    for j in range(project_n):
        for row1, row2 in project_activity[project_list[j]].edges():
            m.addConstr(ST[j][row1] + project_activity[project_list[j]].node[row1]['duration'], GRB.LESS_EQUAL, ST[j][row2],
                        name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))

    #####constrain 10,11
    for j in range(project_n):
        for row1 in project_activity[project_list[j]].nodes():
            for row2 in project_activity[project_list[j]].nodes():
                if row1 != row2 and len(list(set(project_activity[project_list[j]].node[row1]['rk_resources']).intersection(
                        project_activity[project_list[j]].node[row2]['rk_resources']))) > 0:
                    m.addConstr(
                        ST[j][row1] + project_activity[project_list[j]].node[row1]['duration'] - M * (1 - y[j, row1, row2]),
                        GRB.LESS_EQUAL, ST[j][row2],
                        name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                    m.addConstr(
                        ST[j][row2] + project_activity[project_list[j]].node[row2]['duration'] - M * (y[j, row1, row2]),
                        GRB.LESS_EQUAL, ST[j][row1],
                        name="constraint_11_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                    # m.addConstr(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)

    #####constrain 12
    for j in range(project_n):
        for row in project_activity[project_list[j]].nodes():
            m.addConstr(CT[j], GRB.GREATER_EQUAL, ST[j][row] + project_activity[project_list[j]].node[row]['duration'],
                        name="constraint_12_project_%d_activity_%s" % (j, row))

    #####constrain 13
    for j in range(project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + review_duration[j], name="constraint_13_project_%d" % j)

    #####constrain 14
    for i in range(-1, project_n):
        for j in range(project_n):
            if i != j:
                m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - M * (1 - z[i, j]) + review_duration[j],
                            name="constraint_14_project_%d_project_%d" % (i, j))

    #####constrain 15
    for j in range(project_n):
        m.addConstr(quicksum(z[i, j] for i in range(-1, project_n) if i != j), GRB.EQUAL, 1,
                    name="constraint_15_project_%d" % j)

    #####constrain 16
    m.addConstr(quicksum(z[-1, j] for j in range(project_n)), GRB.EQUAL, 1, name="constraint_16")

    #####constrain 17
    for i in range(project_n):
        m.addConstr(quicksum(z[i, j] for j in range(project_n) if j != i), GRB.LESS_EQUAL, 1,
                    name="constraint_17_project_%d" % i)
    m.update()

    # for i in range(project_n):
    #     for j in range(project_n):
    #         if i!=j:
    #             m.addConstr(z[i,j]+z[j,i],GRB.LESS_EQUAL,1)

    # Set optimization objective - minimize sum of
    expr = LinExpr()
    for j in range(project_n):
        expr.add(w[j] * TD[j])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    ##########################################
    m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0
    m.optimize()
    print('project_n=%d' % project_n)
    for j in range(project_n):
        print(len(project_activity[project_list[j]].edges()))

    time_cost = time.clock() - start_time
    print('time cost=', time_cost)
    # Print solution
    m.write(join(output_path, 'original.lp'))
    m.write(join(output_path, 'original.sol'))

    return m.objVal, time_cost
# print decision variables
# print 'review sequence', [row for row in z if z[row].x>=0.98]
# print 'supplier-project-order', [row for row in x if x[row].x>=0.98]
# print 'supplier-prject-quality', [[row,q[row].x] for row in q if q[row].x>0]
# print 'rk_resource_order', [[row,y[row].x] for row in y if y[row].x>0.98]
# print 'ST#########################'
# for j in range(project_n):
#     print [[j,row,ST[j][row].x] for row in ST[j]]
#
# print 'DT[project_id,DT]', [[row,DT[row].x] for row in DT]
# print 'CT[project_id,CT]', [[row,CT[row].x] for row in CT]
# print 'DD[project_id,DD]', [[row,DD[row]] for row in range(project_n)]
# print 'TD[project_id,TD]', [[row,TD[row].x] for row in TD]
if __name__ == "__main__":
    original_model('C:/Users/mteng/Desktop/small case/', 'C:/Users/mteng/Desktop/MIP/')
    (objVal, cost) = original_model('./Inputs/P=10/', 'C:/Users/mteng/Desktop/MIP/')
    print('ObjVal', objVal, 'cost', cost)