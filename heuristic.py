from gurobipy import Model, GRB, quicksum, LinExpr
from input_data import supplier_project_shipping, project_list, project_activity, DD, resource_supplier_capacity, \
    project_n, resource_project_demand, resource_supplier_list, M, c, B, resource_supplier_release_time, \
    review_duration, w
from sys import exit


def optmize_single_project(x, q, j):
    '''
    Given the generated x for single project, try to optimize the tardiness of the project.
    :param x: the assignment of resource supplier to project
    :param j: index of project
    :return:
    '''

    m = Model("SingleProject_%d" % j)

    #### Create variables ####
    project = project_list[j]

    ## Project complete data,Project Tadeness,construction completion time
    CT = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(CT%d)" % j)

    ## Activity start time
    ST = {}
    project_activities = project_activity[project]
    print(project_activities.nodes())
    for row in project_activities.nodes():
        ST[row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(ST1,%s)" % row)

    ## Review sequence z_ij
    ## move to annealing objective function

    # y
    y = {}
    for activity_i in project_activities.nodes():
        for activity_j in project_activities.nodes():
            print(project_activities.node[activity_i])
            print(dir(project_activities.node[activity_i]))
            if activity_i != activity_j and len(list(
                    set(project_activities.node[activity_i]['rk_resources']).intersection(
                        project_activities.node[activity_j]['rk_resources']))) > 0:
                y[activity_i, activity_j] = m.addVar(obj=0, vtype=GRB.BINARY,
                                                     name="(y,%s,%s)" % (activity_i, activity_j))
    m.update()

    #### Create constrains ####
    ## Constrain 2: project complete data>due data
    ## move to annealing objective function

    ## Constrain 3: supplier capacity limit
    ## move to annealing neighbor & random generator

    ## Constrain 4,6: project demand require; each project receive from one supplier for each resource
    ## move to annealing neighbor & random generator

    ## constrain 5: shipping constrain
    ## move to annealing neighbor & random generator

    ## Constrain 7:budget limit
    ## move to annealing constraint valid

    ## Constrain 8: activity starting constrain
    for a in project_activities.nodes():
        for r in project_activities.node[a]['resources']:
            resource_delivered_days = 0
            for s in resource_supplier_list[r]:
                resource_delivered_days += x.get((r, s, project), 0) * \
                                           (resource_supplier_release_time[r, s] + supplier_project_shipping[
                                               r, s, project])
            m.addConstr(resource_delivered_days, GRB.LESS_EQUAL, ST[a])

    ## Constrain 9 activity sequence constrain
    for row1, row2 in project_activities.edges():
        print(row1, '#', row2, '#', j)
        print(ST)
        m.addConstr(ST[row1] + project_activities.node[row1]['duration'], GRB.LESS_EQUAL,
                    ST[row2])

    ## Constrain 10,11
    for row1 in project_activities.nodes():
        for row2 in project_activities.nodes():
            if row1 != row2 and len(list(
                    set(project_activities.node[row1]['rk_resources']).intersection(
                        project_activities.node[row2]['rk_resources']))) > 0:
                m.addConstr(ST[row1] + project_activities.node[row1]['duration'] - M * (
                    1 - y[row1, row2]), GRB.LESS_EQUAL, ST[row2])
                m.addConstr(
                    ST[row2] + project_activities.node[row2]['duration'] - M * (y[row1, row2]),
                    GRB.LESS_EQUAL, ST[row1])
                # m.addConstr(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)

    ## Constrain 12
    for row in project_activities.nodes():
        print(project_activities.node[row]['duration'])
        m.addConstr(CT, GRB.GREATER_EQUAL, ST[row] + project_activities.node[row]['duration'])

    ## Constrain 13
    ## move to anealing objective function

    ## Constrain 14
    ## move to anealing objective function

    ## Constrain 15
    ## move to anealing objective function

    ## Constrain 16
    ## move to anealing objective function

    ## Constrain 17
    ## move to anealing objective function

    m.update()

    # Set optimization objective - minimize completion time
    expr = LinExpr()
    expr.add(CT)
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    ##########################################
    m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0
    m.optimize()

    return m.objVal


def objective_function(x, q):
    '''
    Main Objective function, the sum of tardiness of all the projects.
    :param x: solution is fixed x_sj
    :param q: q_sj is the quantity shipped from supplier s to project j
    :return: The cost and the computed values of all the variables.
    '''
    m = Model("Overall_Model")

    CT = {}
    DT = {}
    TD = {}

    #### Add Variable ####

    for j in range(project_n):
        ## solve individual model get Project complete date
        CT[j] = optmize_single_project(x, q, j)

        ## Project Tadeness,construction completion time
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT%d)" % j)
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(TD%d)" % j)

    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT-1)")

    ## Review Sequence z_ij
    z = {}
    for i in range(project_n):
        for j in range(project_n):
            if i != j:
                z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (i, j))

    for j in range(project_n):
        z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (-1, j))
    m.update();

    #### Add Constraint ####
    ## Constrain 2: project complete data>due data ##
    for j in range(project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, DD[j])

    ## Constraint 13
    for j in range(project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + review_duration[j])

    ## Constraint 14
    for i in range(-1, project_n):
        for j in range(project_n):
            if i != j:
                m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - M * (1 - z[i, j]) + review_duration[j])

    ## Constrain 15
    for j in range(project_n):
        m.addConstr(quicksum(z[i, j] for i in range(-1, project_n) if i != j), GRB.EQUAL, 1)

    ## Constrain 16
    m.addConstr(quicksum(z[-1, j] for j in range(project_n)), GRB.EQUAL, 1)

    ## Constrain 17
    for i in range(project_n):
        m.addConstr(quicksum(z[i, j] for j in range(project_n) if j != i), GRB.LESS_EQUAL, 1)
    m.update()

    # Set optimization objective - minimize sum of
    expr = LinExpr()
    for j in range(project_n):
        expr.add(w[j] * TD[j])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()

    m.params.presolve = 1
    m.update()

    m.optimize()

    return m.objVal


def neighbor(x):
    from random import choice
    all_rsp = list(x.keys())
    all_size = len(all_rsp)
    iterated = set()
    iter_num = 0
    while len(iterated) < all_size:
        (r, s, p) = choice(all_rsp)
        iterated.add((r, s, p))
        print('mute:', r, s, p)
        demand = resource_project_demand[r, p]

        supplier_candidates = [(r, s_, p, resource_supplier_capacity[r_, s_]) \
                               for (r_, s_) in resource_supplier_capacity \
                               if r_ == r and s_ != s
                               and resource_supplier_capacity[r_, s] >= demand]
        if supplier_candidates: break
        iter_num += 1;
    else:
        return x
    # change for previous supplier
    x.pop((r, s, p))
    resource_supplier_capacity[r, s] += demand

    # change for new supplier
    new_supplier = choice(supplier_candidates)[1]
    x[r, new_supplier, p] = 1
    resource_supplier_capacity[r, new_supplier] -= demand
    # print(x)


def acceptance_probability(old_cost, new_cost, T):
    import math
    return math.e ** ((old_cost - new_cost) / T)


def eval_q(x):
    q = {}
    for (i, j, k) in x:
        # i resource, j supplier, k project
        q[i, j, k] = 0 if x[i, j, k] == 0 else resource_project_demand[i, k]
    return q


def constraint_valid(x, q):
    '''
    Check whether constraint valid for solution x
    :param x:
    :return: True if x is valid for constraint False else
    '''

    # Constraint 7
    resource_cost = 0
    for (i, j, k) in q:
        resource_cost += c[i, j, k] * q[i, j, k]
    return resource_cost <= B


def anneal(x_init):
    from random import random
    x = x_init
    q = eval_q(x)
    old_cost = objective_function(x, q)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        i = 1
        while i <= 100:
            # stop until find a feasible solution
            while True:
                x_new = neighbor(x)
                q_new = eval_q(x_new)
                if constraint_valid(x_new, q_new): break

            new_cost = objective_function(x_new, q_new)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                x = x_new
                old_cost = new_cost
            i += 1
        T = T * alpha
    return x, old_cost


def init_x():
    x = {}

    resources = list(set([r for (r, p) in resource_project_demand]))
    resources.sort()

    suppliers = list(set([s for (r, s) in resource_supplier_capacity]))
    suppliers.sort()
    # print(resources)

    for r in resources:
        # demands of resource r for different projects order by demand from high to low
        demands = []
        for p in project_list:
            if resource_project_demand[r, p] > 0:
                demands.append((p, resource_project_demand[r, p]))

        demands.sort(key=lambda x: x[1], reverse=True)
        print('-' * 50)
        print('allocate resource %s' % r)
        print('demands:\n%r' % demands)
        # supply of resource r from different suppliers
        supply = [(s, resource_supplier_capacity[r_, s]) for (r_, s) in resource_supplier_capacity if r_ == r]
        print('supply:\n%r' % supply)

        for (p, demand) in demands:
            print('trying to find %s for %s (%r)' % (r, p, demand))
            print(len(supply), 'all:%r' % supply)
            supplier_candidates = [s for (s, capacity) in supply if
                                   capacity >= demand]

            if not supplier_candidates:
                print('Infeasible')
                exit()
            supplier = choice(supplier_candidates)
            resource_supplier_capacity[r, supplier] -= demand

            x[r, supplier, p] = 1
            print('project {p} choose supplier {s} (capacity {c}) to provide {r} (demand {d})'.format(
                p=p, s=supplier, r=r, c=resource_supplier_capacity[r, supplier], d=demand))
    return x


if __name__ == '__main__':
    from random import choice

    ## Generate x init
    x = init_x()
    print(x)
    print('-' * 100)
    print(neighbor(x))
    # exit(0)
    print('objVal %r resource supplier project arrangement: %r' % anneal(x))
