import logging
import random
import time
from multiprocessing.pool import Pool

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from gurobipy import Model, GRB, quicksum, LinExpr

from input_data import load_data

logging.basicConfig(filename='heuristic_ga.log', level=logging.DEBUG)

_tardiness_obj_trace = []
_last_x = None
_last_CT = None
_pool = None
_delta_trace = []
_CT_map = {}
_delta_dim = 0
_delta_project_idx = {}


def _CT_map_key(project, suppliers):
    sorted_suppliers = sorted(suppliers)
    return '%s_' * (1 + len(sorted_suppliers)) % tuple([project] + sorted_suppliers)


def _get_CT(project, suppliers):
    return _CT_map.get(_CT_map_key(project, suppliers))


def _put_CT(project, suppliers, val):
    _CT_map[_CT_map_key(project, suppliers)] = val


def _get_project_suppliers_map(x, project_list):
    rlt = {p: [] for p in project_list}
    for (r, s, p), v in x.items():
        if v != 0:
            rlt[p].append(s)
        else:
            raise Exception("wrong!")
    return rlt


def _normalize(arr):
    arr_sum = sum(arr)
    for i in range(len(arr)):
        arr[i] = 1. * arr[i] / arr_sum
    return arr


def _random_delta_weight_for_projects(num, Type):
    rlt = Type([1]*num)
    #rlt = Type(np.random.random_sample(num).tolist())
    #rlt = [1]*num
    return _normalize(rlt)


def _mate(ind1, ind2):
    global _delta_trace
    raw = tools.cxTwoPoint(ind1, ind2)
    rlt = [_normalize(e) for e in raw]
    _delta_trace.append(['mate'] + rlt[0])
    _delta_trace.append(['mate'] + rlt[1])
    return rlt


def _mutate(ind, mutate_prob):
    for i in range(len(ind)):
        if random.random() < mutate_prob:
            ind[i] = random.random()
    rlt = _normalize(ind)
    _delta_trace.append(['mutate'] + rlt)
    return rlt,


def _objective_function_for_delta_weight(D, delta_weight):
    m = Model("model_for_supplier_assignment")
    x = {}
    q = {}
    for (r, s, p) in D.supplier_project_shipping:
        # i resource, j supplier, k project
        x[r, s, p] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s_%s" % (r, s, p))
        q[r, s, p] = m.addVar(vtype=GRB.CONTINUOUS, name="q_%s_%s_%s" % (r, s, p))

    AT = {}

    for j in range(D.project_n):
        for k in sorted([r for r, p in D.resource_project_demand if p == D.project_list[j]]):
            AT[j, k] = m.addVar(vtype=GRB.CONTINUOUS, name="AT_%s_%s" % (j, k))

    m.update()

    ## define constraints
    # constraint 20(3)
    for (r, s) in D.resource_supplier_capacity:
        m.addConstr(quicksum(q[r, s, D.project_list[j]] for j in range(D.project_n)), GRB.LESS_EQUAL,
                    D.resource_supplier_capacity[r, s],
                    name="constraint_3_resource_%s_supplier_%s" % (r, s))

    # constraint 21(4) 23(6)
    for (r, p) in D.resource_project_demand:
        m.addConstr(quicksum(x[r, i, p] for i in D.resource_supplier_list[r]), GRB.EQUAL, 1,
                    name="constraint_6_resource_%s_project_%s" % (r, p))
        m.addConstr(quicksum(q[r, i, p] for i in D.resource_supplier_list[r]), GRB.GREATER_EQUAL,
                    D.resource_project_demand[r, p], name="constraint_4_resource_%s_project_%s" % (r, p))

    # constraint 22(5)
    for (i, j, k) in q:
        # i resource, j supplier, k project
        m.addConstr(q[i, j, k], GRB.LESS_EQUAL, D.M * x[i, j, k],
                    name="constraint_5_resource_%s_supplier_%s_project_%s" % (i, j, k))

    # constraint 7
    expr = LinExpr()
    for (i, j, k) in q:
        expr = expr + D.c[i, j, k] * q[i, j, k]
    m.addConstr(expr, GRB.LESS_EQUAL, D.B, name="constraint_7")

    # constraint 8
    for j in range(D.project_n):
        p = D.project_list[j]
        project_resources = sorted([r for (r, p_) in D.resource_project_demand.keys() if p_ == p])
        for r in project_resources:
            suppliers = D.resource_supplier_list[r]

            #print(list(D.supplier_project_shipping.keys())[:10])
            # print(D.supplier_project_shipping['NK0g77', 'S1671', 'P1'])
            #print(list(x.keys())[:10])
            # print(x['NK0g77', 'S1671', 'P1'])
            m.addConstr(
                quicksum(
                    x[r, s, p] * (D.resource_supplier_release_time[r, s] + D.supplier_project_shipping[r, s, p]) for
                    s in
                    suppliers), GRB.LESS_EQUAL, AT[j, r],
                name="constraint_8_project_%d_resource_%s_deliver" % (j, r))

    m.update()

    expr = LinExpr()
    for j in range(D.project_n):
        for r in sorted([r for (r, p_) in D.resource_project_demand.keys() if p_ == p]):
            expr.add(delta_weight[_delta_project_idx[j, r]] * AT[j, r])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    ##########################################
    m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0
    m.optimize()

    X_ = {}
    for (i, j, k) in D.supplier_project_shipping:
        v = m.getVarByName("x_%s_%s_%s" % (i, j, k))
        if v.X == 1:
            X_[i, j, k] = 1

    return -_objective_function_for_tardiness(X_, D),


def optmize_single_project(x, j, project_list, project_activity, resource_supplier_list, resource_supplier_release_time,
                           supplier_project_shipping, M):
    m = Model("SingleProject_%d" % j)

    #### Create variables ####
    project = project_list[j]

    ## Project complete data,Project Tadeness,construction completion time
    CT = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(CT%d)" % j)

    ## Activity start time
    ST = {}
    project_activities = project_activity[project]
    # print(project_activities.nodes())
    for row in project_activities.nodes():
        ST[row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(ST%d,%s)" % (j, row))

    ## Review sequence z_ij
    ## move to annealing objective function

    # y
    y = {}
    for activity_i in project_activities.nodes():
        for activity_j in project_activities.nodes():
            # print(project_activities.node[activity_i])
            # print(dir(project_activities.node[activity_i]))
            if activity_i != activity_j and len(list(
                    set(project_activities.node[activity_i]['rk_resources']).intersection(
                        project_activities.node[activity_j]['rk_resources']))) > 0:
                y[activity_i, activity_j] = m.addVar(obj=0, vtype=GRB.BINARY,
                                                     name="(y%d,%s,%s)" % (j, activity_i, activity_j))
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
                                           (resource_supplier_release_time[r, s] +
                                            supplier_project_shipping[
                                                r, s, project])
            m.addConstr(resource_delivered_days, GRB.LESS_EQUAL, ST[a],
                        name="constraint_8_project_%d_activity_%s_resource_%s" % (j, a, r))

    ## Constrain 9 activity sequence constrain
    for row1, row2 in project_activities.edges():
        # print(row1, '#', row2, '#', j)
        # print(ST)
        m.addConstr(ST[row1] + project_activities.node[row1]['duration'], GRB.LESS_EQUAL,
                    ST[row2], name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))

    ## Constrain 10,11
    for row1 in project_activities.nodes():
        for row2 in project_activities.nodes():
            if row1 != row2 and len(list(
                    set(project_activities.node[row1]['rk_resources']).intersection(
                        project_activities.node[row2]['rk_resources']))) > 0:
                m.addConstr(ST[row1] + project_activities.node[row1]['duration'] - M * (
                    1 - y[row1, row2]), GRB.LESS_EQUAL, ST[row2],
                            name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                m.addConstr(
                    ST[row2] + project_activities.node[row2]['duration'] - M * (y[row1, row2]),
                    GRB.LESS_EQUAL, ST[row1],
                    name="constraint_11_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                # m.addConstr(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)

    ## Constrain 12
    for row in project_activities.nodes():
        m.addConstr(CT, GRB.GREATER_EQUAL, ST[row] + project_activities.node[row]['duration'],
                    name="constraint_12_project_%d_activity_%s" % (j, row))

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
    # m.write(join(output_dir, "heuristic_%d.lp" % j))
    # m.write(join(output_dir, "heuristic_%d.sol" % j))
    logging.info("project %d with optimalVal %r" % (j, m.objVal))
    return m.objVal


def _get_project_to_recompute(x, project_n, project_list):
    if _last_x is None:
        return range(project_n), []
    # logging.info('x keys:%r' % x.keys())
    # logging.info('last x keys:%r' % _last_x.keys())
    diffs_proj = set(p for r, s, p in (x.keys() ^ _last_x.keys()))
    logging.info('diffs_proj:%r' % diffs_proj)
    diffs_proj_idx = sorted([project_list.index(p) for p in diffs_proj])
    same_proj_idx = [i for i in range(project_n) if project_list[i] not in diffs_proj]
    logging.info("project    recompute idx:%r" % diffs_proj_idx)
    logging.info("project no-recompute idx:%r" % same_proj_idx)
    return diffs_proj_idx, same_proj_idx


def _objective_function_for_tardiness(x, D):
    global _last_CT, _last_x, _pool
    m = Model("Overall_Model")

    CT = {}
    CT_ASYNC = dict()
    # project_to_recompute, project_no_recompute = get_project_to_recompute(x, D.project_n, D.project_list)
    project_suppliers = _get_project_suppliers_map(x, D.project_list)
    for j in range(D.project_n):
        p = D.project_list[j]
        history_CT = _get_CT(p, project_suppliers[p])
        if history_CT is None:
            CT_ASYNC[j] = _pool.apply_async(optmize_single_project,
                                            (x, j, D.project_list, D.project_activity, D.resource_supplier_list,
                                             D.resource_supplier_release_time,
                                             D.supplier_project_shipping, D.M))
        else:
            # logging.info('%s [%s]' % (p, ','.join(sorted(project_suppliers[p]))))
            CT[j] = history_CT
            logging.info('project %s get historical value %f.' % (p, history_CT))

    for j in CT_ASYNC:
        CT[j] = CT_ASYNC[j].get()
        p = D.project_list[j]
        _put_CT(p, project_suppliers[p], CT[j])

    # _last_CT = deepcopy(CT)
    # _last_x = deepcopy(x)

    # if len(project_no_recompute) == D.project_n:
    #     return _tardiness_obj_trace[-1]

    # self.last_CT = deepcopy(CT)
    DT = {}
    TD = {}
    for j in range(D.project_n):
        ## Project Tadeness,construction completion time
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT%d)" % j)
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(TD%d)" % j)

    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT-1)")

    ## Review Sequence z_ij
    z = {}
    for i in range(D.project_n):
        for j in range(D.project_n):
            if i != j:
                z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (i, j))

    for j in range(D.project_n):
        z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (-1, j))
    m.update();

    #### Add Constraint ####
    ## Constrain 2: project complete data>due data ##
    for j in range(D.project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, D.DD[j], name="constraint_2_project_%d" % j)

    ## Constraint 13
    for j in range(D.project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + D.review_duration[j], name="constraint_13_project_%d" % j)

    ## Constraint 14
    for i in range(-1, D.project_n):
        for j in range(D.project_n):
            if i != j:
                m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - D.M * (1 - z[i, j]) + D.review_duration[j],
                            name="constraint_14_project_%d_project_%d" % (i, j))

    ## Constrain 15
    for j in range(D.project_n):
        m.addConstr(quicksum(z[i, j] for i in range(-1, D.project_n) if i != j), GRB.EQUAL, 1,
                    name="constraint_15_project_%d" % j)

    ## Constrain 16
    m.addConstr(quicksum(z[-1, j] for j in range(D.project_n)), GRB.EQUAL, 1, name="constraint_16")

    ## Constrain 17
    for i in range(D.project_n):
        m.addConstr(quicksum(z[i, j] for j in range(D.project_n) if j != i), GRB.LESS_EQUAL, 1,
                    name="constraint_17_project_%d" % i)
    m.update()

    # Set optimization objective - minimize sum of
    expr = LinExpr()
    for j in range(D.project_n):
        expr.add(D.w[j] * TD[j])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()

    m.params.presolve = 1
    m.update()

    m.optimize()
    # m.write(join(self.output_dir, "heuristic_whole.lp"))
    # m.write(join(self.output_dir, "heuristic_whole.sol"))

    # self.obj_value_trace.append(m.objVal)
    _tardiness_obj_trace.append(m.objVal)
    logging.info("Tardiness value:%r" % m.objVal)
    return m.objVal


def heuristic_ga_optimize(input_path, out_path):
    start = time.clock()
    global _last_x, _last_CT, _pool, _delta_trace, _delta_dim, _delta_project_idx
    _tardiness_obj_trace.clear()
    _delta_trace.clear()
    _delta_project_idx.clear()
    _CT_map.clear()
    _last_x = None
    _last_CT = None
    _pool = Pool(5)

    D = load_data(input_path)

    # initialization for GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    _delta_dim = 0
    for j in range(D.project_n):
        p = D.project_list[j]
        for r in sorted([r_ for (r_, p_) in D.resource_project_demand.keys() if p_ == p]):
            _delta_project_idx[j, r] = _delta_dim
            _delta_dim += 1

    toolbox.register("individual", _random_delta_weight_for_projects, _delta_dim, creator.Individual)
    toolbox.register("population", tools.initRepeat, creator.Individual, toolbox.individual)
    toolbox.register("evaluate", _objective_function_for_delta_weight, D)
    toolbox.register("mate", _mate)
    toolbox.register("mutate", _mutate, mutate_prob=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # print()

    pop = toolbox.population(n=1)
    hof = tools.HallOfFame(1)

    # print(toolbox.individual())
    # print(pop)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, halloffame=hof, verbose=True)
    # print(min(_tardiness_obj_trace), '\n', max(_tardiness_obj_trace))
    # print(_tardiness_obj_trace)
    # logging.info('min tardiness obj trace %r \n max tardiness obj trace:%r\n' % (
    #     min(_tardiness_obj_trace), max(_tardiness_obj_trace)))
    # logging.info(_tardiness_obj_trace)
    return min(_tardiness_obj_trace), time.clock() - start


if __name__ == '__main__':
    # C:/Users/mteng/Desktop/small case/
    # ./Inputs/P=10
    # ./Inputs/case1
    print(heuristic_ga_optimize('./Inputs/P=10', None))
    # d = pd.DataFrame(_delta_trace)
    # d.to_csv('delta_trace.csv')
