from gurobipy import *


def merge(paths, output_path='./merged_solution.sol'):
    with open(output_path, 'w') as f_out:
        for p in paths:
            with open(p, 'r') as f_in:
                for line in f_in:
                    if not line.startswith('#'):
                        f_out.writelines(line + '\n')


def _read_vars(path):
    variables = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#') and not line.startswith('\n'):
                splits = line.split(' ')
                #print(splits)
                k, v = splits[0], splits[1]
                variables[k] = float(v)
    return variables


def _close_enough(a, b):
    return abs(a - b) < 1e-5


def eval(expr, variables):
    if isinstance(expr, (int, float, complex)):
        return expr
    elif isinstance(expr, Var):
        return variables[expr.VarName]
    val = 0
    for i in range(expr.size()):
        # print('eval variable:', expr.getVar(i).VarName)
        val += variables[expr.getVar(i).VarName] * expr.getCoeff(i)
    val += expr.getConstant()
    return val


def _verify_expression(left, sense, right, name):
    global variables
    left_val = eval(left, variables)
    right_val = eval(right, variables)
    if sense == GRB.LESS_EQUAL:
        valid = left_val <= right_val or _close_enough(left_val - right_val, 0)
    elif sense == GRB.EQUAL:
        valid = _close_enough(left_val, right_val)
    else:
        valid = left_val >= right_val or _close_enough(left_val - right_val, 0)
    if not valid:
        raise Exception("%s is invalid." % name)


def verify(path, D):
    global variables
    supplier_project_shipping, project_list, project_activity, DD, resource_supplier_capacity, \
    project_n, resource_project_demand, resource_supplier_list, M, c, B, resource_supplier_release_time, \
    review_duration, w = D

    variables = _read_vars(path)
    m = Model('construction')
    # m.setParam('OutputFlag', False)
    ##############################################################
    # Create variables############################################
    #####supplier-project shipping decision x and shipping quality
    x = {}
    q = {}
    for (i, j, k) in supplier_project_shipping:
        # i resource, j supplier, k project
        x[i, j, k] = m.addVar(obj=0, vtype=GRB.BINARY, name="x_%s_%s_%s" % (i, j, k))
        q[i, j, k] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="q_%s_%s_%s" % (i, j, k))
    # print('add var x,q')
    #####Project complete data,Project Tadeness,construction completion time
    DT = {}
    TD = {}
    CT = {}
    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_-1")  # project start time
    for j in range(project_n):
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_%d" % j)  # project j complete time
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="TD_%d" % j)  # project j complete time
        CT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="CT_%d" % j)  # project j complete time
    # print('add var DT TD CT')
    #####Activity start time
    ST = []
    for j in range(project_n):
        ST.append({})
        for row in project_activity[project_list[j]].nodes():
            ST[j][row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="ST_%d_%s" % (j, row))
    # print('add var ST')
    #####Review sequence
    z = {}
    for i in range(project_n):
        for j in range(project_n):
            if i != j:
                z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="z_%d_%d" % (i, j))

    for j in range(project_n):
        z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="z_%d_%d" % (-1, j))
    # print('add var z')
    #####
    y = {}
    for j in range(project_n):
        for row1 in project_activity[project_list[j]].nodes():
            for row2 in project_activity[project_list[j]].nodes():
                # print project_activity[project_list[j]].node[row1]
                if row1 != row2 and len(
                        list(set(project_activity[project_list[j]].node[row1]['rk_resources']).intersection(
                            project_activity[project_list[j]].node[row2]['rk_resources']))) > 0:
                    y[j, row1, row2] = m.addVar(obj=0, vtype=GRB.BINARY, name="y_%d_%s_%s" % (j, row1, row2))

    m.update()
    for j in range(D.project_n):
        _verify_expression(DT[j] - TD[j], GRB.LESS_EQUAL, DD[j], name="constraint_2_project_%d" % j)
    # print('add constr 2')
    ##### constrain 3: supplier capacity limit
    for (row1, row2) in resource_supplier_capacity:
        _verify_expression(quicksum(q[row1, row2, project_list[j]] for j in range(project_n)), GRB.LESS_EQUAL,
                           resource_supplier_capacity[row1, row2],
                           name="constraint_3_resource_%s_supplier_%s" % (row1, row2))
    # print('add constr 3')
    #####constrain 4,6: project demand require; each project receive from one supplier for each resource
    for (row1, row2) in resource_project_demand:
        _verify_expression(quicksum(x[row1, i, row2] for i in resource_supplier_list[row1]), GRB.EQUAL, 1,
                           name="constraint_6_resource_%s_project_%s" % (row1, row2))
        _verify_expression(quicksum(q[row1, i, row2] for i in resource_supplier_list[row1]), GRB.GREATER_EQUAL,
                           resource_project_demand[row1, row2],
                           name="constraint_4_resource_%s_project_%s" % (row1, row2))
    # print('add constr 4,6')
    #####constrain 5: shipping constrain
    for (i, j, k) in q:
        # i resource, j supplier, k project
        _verify_expression(q[i, j, k], GRB.LESS_EQUAL, M * x[i, j, k],
                           name="constraint_5_resource_%s_supplier_%s_project_%s" % (i, j, k))
    # print('add constr 5')
    #####constrain 7:budget limit
    expr = LinExpr()
    for (i, j, k) in q:
        expr.addTerms(c[i, j, k], q[i, j, k])
    _verify_expression(expr, GRB.LESS_EQUAL, B, name="constraint_7")
    # print('add constr 7')
    #####constrain 8: activity starting constrain
    for j in range(project_n):
        for row in project_activity[project_list[j]].nodes():
            for row1 in project_activity[project_list[j]].node[row]['resources']:
                _verify_expression(quicksum(x[row1, i, project_list[j]] * (
                    resource_supplier_release_time[row1, i] + supplier_project_shipping[row1, i, project_list[j]]) for i
                                            in
                                            resource_supplier_list[row1]), GRB.LESS_EQUAL, ST[j][row],
                                   name="constraint_8_project_%d_activity_%s_resource_%s" % (j, row, row1))
    # print('add constr 8')
    #####constrain 9 activity sequence constrain
    for j in range(project_n):
        for row1, row2 in project_activity[project_list[j]].edges():
            _verify_expression(ST[j][row1] + project_activity[project_list[j]].node[row1]['duration'], GRB.LESS_EQUAL,
                               ST[j][row2],
                               name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))
    # print('add constr 9')
    #####constrain 10,11
    for j in range(project_n):
        for row1 in project_activity[project_list[j]].nodes():
            for row2 in project_activity[project_list[j]].nodes():
                if row1 != row2 and len(
                        list(set(project_activity[project_list[j]].node[row1]['rk_resources']).intersection(
                            project_activity[project_list[j]].node[row2]['rk_resources']))) > 0:
                    _verify_expression(
                        ST[j][row1] + project_activity[project_list[j]].node[row1]['duration'] - M * (
                            1 - y[j, row1, row2]),
                        GRB.LESS_EQUAL, ST[j][row2],
                        name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                    _verify_expression(
                        ST[j][row2] + project_activity[project_list[j]].node[row2]['duration'] - M * (y[j, row1, row2]),
                        GRB.LESS_EQUAL, ST[j][row1],
                        name="constraint_11_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                    # _verify_expression(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)
    # print('add constr 10 11')
    #####constrain 12
    for j in range(project_n):
        for row in project_activity[project_list[j]].nodes():
            _verify_expression(CT[j], GRB.GREATER_EQUAL,
                               ST[j][row] + project_activity[project_list[j]].node[row]['duration'],
                               name="constraint_12_project_%d_activity_%s" % (j, row))
    # print('add constr 12')
    #####constrain 13
    for j in range(project_n):
        _verify_expression(DT[j], GRB.GREATER_EQUAL, CT[j] + review_duration[j], name="constraint_13_project_%d" % j)
    # print('add constr 13')
    #####constrain 14
    for i in range(-1, project_n):
        for j in range(project_n):
            if i != j:
                _verify_expression(DT[j], GRB.GREATER_EQUAL, DT[i] - M * (1 - z[i, j]) + review_duration[j],
                                   name="constraint_14_project_%d_project_%d" % (i, j))
    # print('add constr 14')
    #####constrain 15
    for j in range(project_n):
        _verify_expression(quicksum(z[i, j] for i in range(-1, project_n) if i != j), GRB.EQUAL, 1,
                           name="constraint_15_project_%d" % j)
    # print('add constr 15')
    #####constrain 16
    _verify_expression(quicksum(z[-1, j] for j in range(project_n)), GRB.EQUAL, 1, name="constraint_16")
    # print('add constr 16')
    #####constrain 17
    for i in range(project_n):
        _verify_expression(quicksum(z[i, j] for j in range(project_n) if j != i), GRB.LESS_EQUAL, 1,
                           name="constraint_17_project_%d" % i)
    # print('add constr 17')
    m.update()

    # for i in range(project_n):
    #     for j in range(project_n):
    #         if i!=j:
    #             _verify_expression(z[i,j]+z[j,i],GRB.LESS_EQUAL,1)

    # Set optimization objective - minimize sum of
    expr = LinExpr()
    for j in range(project_n):
        expr.addTerms(w[j], TD[j])
    # print('add obj')
    m.setObjective(expr, GRB.MINIMIZE)


if __name__ == '__main__':
    from os import listdir
    from input_data import load_data

    # fs = ['heuristic/%s'%i for i in listdir('heuristic') if i.endswith('sol')]
    # merge(fs)
    #verify('./merged_solution.sol', load_data('./Inputs/P=15'))
    variables = _read_vars('./merged_solution.sol')
    print(variables['DT_15'], variables['CT_15'], variables['DT_15'] - variables['CT_15'])
    # print(variables['ST_0_T012'], variables['ST_0_T022'], variables['ST_0_T012'] - variables['ST_0_T022'])
