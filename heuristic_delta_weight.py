import logging
import time
from multiprocessing.pool import Pool

from gurobipy import Model, GRB, quicksum, LinExpr

from input_data import load_data
from os.path import exists
from os import makedirs
from os.path import join

logging.basicConfig(filename='my.log', level=logging.INFO)

_tardiness_obj_trace = []
_last_x = None
_last_CT = None
_pool = None
_delta_trace = []
_CT_map = {}
_historical_delta_weight_idx_map = {}
_output_path = './heuristic/'


def _time_cal(str, _time_call_trace=[]):
    now = time.clock()
    if _time_call_trace:
        logging.info("%s cost %.2f" % (str, now - _time_call_trace[-1]))
        _time_call_trace.pop(0)
    _time_call_trace.append(now)


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        return result

    return timed


def _CT_map_key(project, suppliers):
    sorted_suppliers = sorted(suppliers)
    return '%s_' * (1 + len(sorted_suppliers)) % tuple([project] + sorted_suppliers)


def _get_CT(project, suppliers):
    return _CT_map.get(_CT_map_key(project, suppliers))


def _put_CT(project, suppliers, val):
    _CT_map[_CT_map_key(project, suppliers)] = val


def _get_historical_delta_weight_idx_map(project, suppliers):
    return _historical_delta_weight_idx_map.get(_CT_map_key(project, suppliers))


def _put_historical_delta_weight_idx_map(project, suppliers, val):
    _historical_delta_weight_idx_map[_CT_map_key(project, suppliers)] = val


def _get_project_suppliers_map(x, project_list):
    rlt = {p: [] for p in project_list}
    for (r, s, p), v in x.items():
        if v != 0:
            rlt[p].append(s)
        else:
            raise Exception("wrong!")
    return rlt


def _normalize(d):
    total = sum(d.values())
    for k, v in d.items():
        d[k] = 1. * v / total
    pass


def _objective_function_for_delta_weight(D, delta_weight, d1, d2):
    m = Model("model_for_supplier_assignment")
    m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7
    x = {}
    q = {}
    for (r, s, p) in D.supplier_project_shipping:
        x[r, s, p] = m.addVar(vtype=GRB.BINARY, name="x_%s_%s_%s" % (r, s, p))
        q[r, s, p] = m.addVar(vtype=GRB.CONTINUOUS, name="q_%s_%s_%s" % (r, s, p))

    AT = {}
    for j in range(D.project_n):
        for k in [r for r, p in D.resource_project_demand if p == D.project_list[j]]:
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
        expr.addTerms(D.c[i, j, k], q[i, j, k])
    m.addConstr(expr, GRB.LESS_EQUAL, D.B, name="constraint_7")

    # constraint 8
    for j in range(D.project_n):
        p = D.project_list[j]
        project_resources = [r for (r, p_) in D.resource_project_demand.keys() if p_ == p]
        for r in project_resources:
            suppliers = D.resource_supplier_list[r]
            m.addConstr(
                quicksum(
                    x[r, s, p] * (D.resource_supplier_release_time[r, s] + D.supplier_project_shipping[r, s, p]) for
                    s in
                    suppliers), GRB.LESS_EQUAL, AT[j, r],
                name="constraint_8_project_%d_resource_%s_deliver" % (j, r))
    m.update()

    expr = LinExpr()
    for j in range(D.project_n):
        for r in [r for (r, p_) in D.resource_project_demand.keys() if p_ == p]:
            expr.add(delta_weight[j, r] * AT[j, r])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    ##########################################
    m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0
    m.optimize()
    m.write(join(_output_path, 'delta_weight.sol'))
    m.write(join(_output_path, 'delta_weight.lp'))
    X_ = {}
    for (i, j, k) in D.supplier_project_shipping:
        v = m.getVarByName("x_%s_%s_%s" % (i, j, k))
        if v.X == 1:
            X_[i, j, k] = 1

    AT_ = {}
    for j, r in AT:
        val = AT[j, r].X
        if val > 0:
            AT_[j, r] = val

    tardiness_obj_val, skj, sj = _objective_function_for_tardiness(X_, AT_, D)
    new_delta_weight = {}
    # delta_weight_keys = list(delta_weight.keys())
    # delta_weight_keys.sort(key=lambda x: x[1])
    # delta_weight_keys.sort(key=lambda x: x[0])
    for j, r in delta_weight.keys():
        new_delta_weight[j, r] = delta_weight[j, r] * (1 + d1 * (d2 + sj.get(j, 0)) * skj.get((j, r), 0))
        # new_delta_weight[j, r] = 1
    _normalize(new_delta_weight)
    return new_delta_weight


def _get_y_for_activities(y, a1, a2):
    if (a1, a2) in y:
        return y[a1, a2].X
    else:
        return 0


def _sensitivity_for_constraints(AT, j, project, y_, project_activity, M):
    m = Model("SingleProject_%d_for_sensitivity" % j)
    m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7

    ## Project complete data,Project Tadeness,construction completion time
    CT = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="CT_%d" % j)

    ## Activity start time
    ST = {}
    project_activities = project_activity[project]
    for row in project_activities.nodes():
        ST[row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="ST_%d_%s" % (j, row))

    ## Review sequence z_ij
    ## move to annealing objective function

    m.update()

    ## Constrain 8: activity starting constrain
    for a in project_activities.nodes():
        for r in project_activities.node[a]['resources']:
            m.addConstr(ST[a], GRB.GREATER_EQUAL, AT[j, r],
                        name="constraint_8_project_%d_activity_%s_resource_%s" % (j, a, r))

    ## Constrain 9 activity sequence constrain
    for row1, row2 in project_activities.edges():
        m.addConstr(ST[row1] + project_activities.node[row1]['duration'], GRB.LESS_EQUAL,
                    ST[row2], name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))

    ## Constrain 10,11
    for row1 in project_activities.nodes():
        for row2 in project_activities.nodes():
            if row1 != row2 and len(list(
                    set(project_activities.node[row1]['rk_resources']).intersection(
                        project_activities.node[row2]['rk_resources']))) > 0:
                m.addConstr(ST[row1] + project_activities.node[row1]['duration'] - M * (
                    1 - _get_y_for_activities(y_, row1, row2)), GRB.LESS_EQUAL, ST[row2],
                            name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                m.addConstr(
                    ST[row2] + project_activities.node[row2]['duration'] - M * _get_y_for_activities(y_, row1, row2),
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
    m.setParam(GRB.Param.Method, 0)
    m.update()
    # Solve
    # m.params.presolve=0

    m.optimize()

    _skja = {}
    for c in m.getConstrs():
        if c.ConstrName.startswith('constraint_8_project'):
            splits = c.ConstrName.split('_')
            r = splits[7]
            if r not in _skja:
                _skja[r] = []
            _skja[r].append(c.Pi)

            if c.Pi != 0:
                logging.debug('project %d binding resource:%s Pi:%.4g' % (j, splits[-1], c.Pi))
            else:
                logging.debug('project %d not binding resource:%s Pi:%.4g' % (j, splits[-1], c.Pi))
    _skj = {}
    for r in _skja:
        _skj[j, r] = max(_skja[r])
    return _skj


def optimize_single_project(AT, j, project_list, project_activity, M):
    m = Model("SingleProject_%d" % j)
    m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7

    #### Create variables ####
    project = project_list[j]

    ## Project complete data,Project Tadeness,construction completion time
    CT = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="CT_%d" % j)

    ## Activity start time
    ST = {}
    project_activities = project_activity[project]
    for row in project_activities.nodes():
        ST[row] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="ST_%d_%s" % (j, row))

    ## Review sequence z_ij
    ## move to annealing objective function

    # y
    y = {}
    for activity_i in project_activities.nodes():
        for activity_j in project_activities.nodes():
            if activity_i != activity_j and len(list(
                    set(project_activities.node[activity_i]['rk_resources']).intersection(
                        project_activities.node[activity_j]['rk_resources']))) > 0:
                y[activity_i, activity_j] = m.addVar(obj=0, vtype=GRB.BINARY,
                                                     name="y_%d_%s_%s" % (j, activity_i, activity_j))
    m.update()

    ## Constrain 8: activity starting constrain
    for a in project_activities.nodes():
        for r in project_activities.node[a]['resources']:
            m.addConstr(AT[j, r], GRB.LESS_EQUAL, ST[a],
                        name="constraint_8_project_%d_activity_%s_resource_%s" % (j, a, r))

    ## Constrain 9 activity sequence constrain
    for row1, row2 in project_activities.edges():
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
    m.write(join(_output_path, "heuristic_%d.lp" % j))
    m.write(join(_output_path, "heuristic_%d.sol" % j))
    # logging.info("project %d with optimalVal %r" % (j, m.objVal))
    # m.fixedModel()

    skj = _sensitivity_for_constraints(AT, j, project, y, project_activity, M)
    # for c in m.getConstrs():
    #     if c.ConstrName.startswith('constraint_8_project'):
    #         splits = c.ConstrName.split('_')
    #         if c.Pi == 0:
    #             logging.info('project %d bind resource:%s Slack:%.4g'%(j, splits[-1],c.Pi))
    #             break
    # else:
    #     logging.info('project %d not bind'%j)

    return m.objVal, skj


def _sensitivity_analysis_for_tardiness(z, CT, D):
    m = Model("model_for_sensitivity_analysis_for_tardiness")
    m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7
    DT = {}
    TD = {}
    for j in range(D.project_n):
        ## Project Tadeness,construction completion time
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_%d" % j)
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="TD_%d" % j)

    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_-1")
    m.update();

    #### Add Constraint ####
    ## Constrain 2: project complete data>due data ##
    for j in range(D.project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, D.DD[j], name="constraint_2_project_%d" % j)

    ## Constraint 13
    for j in range(D.project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + D.review_duration[j],
                    name="constraint_13_project_%d" % j)

    ## Constraint 14
    for i in range(-1, D.project_n):
        for j in range(D.project_n):
            if i != j:
                m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - D.M * (1 - z[i, j]) + D.review_duration[j],
                            name="constraint_14_project_%d_project_%d" % (i, j))

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

    # logging.info("mm binding info:")
    sj = {}
    for c in m.getConstrs():
        if c.ConstrName.startswith('constraint_13'):
            j = int(c.ConstrName.split('_')[-1])
            if c.Pi != 0:
                # sj[j] = 1
                # logging.info('%s binding Pi:%.4g' % (c.ConstrName, c.Pi))
                pass
            else:
                # sj[j] = 0
                # logging.info('%s not binding Pi:%.4g' % (c.ConstrName, c.Pi))
                pass
            sj[j] = c.Pi
    return sj


def _objective_function_for_tardiness(x, AT, D):
    global _last_CT, _last_x, _pool
    m = Model("Overall_Model")
    m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7

    CT = {}
    CT_ASYNC = dict()
    # project_to_recompute, project_no_recompute = get_project_to_recompute(x, D.project_n, D.project_list)
    project_suppliers = _get_project_suppliers_map(x, D.project_list)
    critical_project_resource = dict()
    for j in range(D.project_n):
        p = D.project_list[j]
        #history_CT = _get_CT(p, project_suppliers[p])
        CT_ASYNC[j] = _pool.apply_async(optimize_single_project,
                                        (AT, j, D.project_list, D.project_activity, D.M))
        # if history_CT is None:
        #     CT_ASYNC[j] = _pool.apply_async(optimize_single_project,
        #                                     (AT, j, D.project_list, D.project_activity, D.M))
        #
        # else:
        #     CT[j] = history_CT
        #     critical_project_resource.update(_get_historical_delta_weight_idx_map(p, project_suppliers[p]))

    for j in CT_ASYNC:
        CT[j], skj = CT_ASYNC[j].get()
        _put_CT(p, project_suppliers[p], CT[j])
        _put_historical_delta_weight_idx_map(p, project_suppliers[p], skj)
        critical_project_resource.update(skj)

    DT = {}
    TD = {}
    for j in range(D.project_n):
        ## Project Tadeness,construction completion time
        DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_%d" % j)
        TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="TD_%d" % j)

    DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="DT_-1")

    ## Review Sequence z_ij
    z = {}
    for i in range(D.project_n):
        for j in range(D.project_n):
            if i != j:
                z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="z_%d_%d" % (i, j))

    for j in range(D.project_n):
        z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="z_%d_%d" % (-1, j))
    m.update();

    #### Add Constraint ####
    ## Constrain 2: project complete data>due data ##
    for j in range(D.project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, D.DD[j], name="constraint_2_project_%d" % j)

    ## Constraint 13
    for j in range(D.project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + D.review_duration[j],
                    name="constraint_13_project_%d" % j)

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
    m.write(join(_output_path, 'tardiness.sol'))
    m.write(join(_output_path, 'tardiness.lp'))
    z_ = {}
    for i, j in z:
        z_[i, j] = z[i, j].X

    critical_projs = _sensitivity_analysis_for_tardiness(z_, CT, D)

    _tardiness_obj_trace.append(m.objVal)

    return m.objVal, critical_project_resource, critical_projs


def heuristic_delta_weight(input_path, output_path=None, converge_count=2, tolerance=1, d1=100, d2=0):
    '''

    :param input_path: the path for the folder of the input files
    :param converge_count: the process will stop when the optimal solution isn't update in converge_count rounds.
    :param tolerance: when abs(last_optimal-current_optimal)<tolerance, the solution are considered as unchanged(converged).
    :param d1: parameter in formula 40
    :param d2: parameter in formula 40
    :return: (objective_value, time_cost) will be returned
    '''
    start = time.clock()
    global _last_x, _last_CT, _pool, _delta_trace, _historical_delta_weight_idx_map, _output_path
    _tardiness_obj_trace.clear()
    _delta_trace.clear()
    _CT_map.clear()
    _historical_delta_weight_idx_map.clear()
    _last_x = None
    _last_CT = None
    _pool = Pool(5)
    if output_path is not None:
        _output_path = output_path
    if not exists(_output_path):
        makedirs(_output_path)
    D = load_data(input_path)

    # initialization for GA
    delta_weight = {}
    for j in range(D.project_n):
        p = D.project_list[j]
        for r in sorted([r_ for (r_, p_) in D.resource_project_demand.keys() if p_ == p]):
            delta_weight[j, r] = 1

    _normalize(delta_weight)

    optimal = 1e10
    current_converge_count = 0
    while current_converge_count < converge_count:
        delta_weight = _objective_function_for_delta_weight(D, delta_weight, d1, d2)
        if _tardiness_obj_trace[-1] < optimal:
            if abs(_tardiness_obj_trace[-1] - optimal) <= tolerance:
                current_converge_count += 1
            else:
                current_converge_count = 0
            optimal = min(optimal, _tardiness_obj_trace[-1])
        else:
            current_converge_count += 1

        print("trace:", _tardiness_obj_trace)
        # break
        # print("current_converge_count:", current_converge_count)
        # print("delta size:", len(delta_weight))
        # print(delta_weight)

    return min(_tardiness_obj_trace), time.clock() - start


if __name__ == '__main__':
    # C:/Users/mteng/Desktop/small case/
    # ./Inputs/P=10
    # ./Inputs/case1

    ### run single
    # data_path = './Inputs/P=15'
    # (objVal, cost) = heuristic_delta_weight(data_path, converge_count=2, tolerance=1, d1=100, d2=0)
    # print(objVal, cost)

    ### run in batch
    import pandas as pd

    d = pd.DataFrame(columns=["Project Size", "Objective Value", "Time Cost"])
    d_idx = 0
    for i in range(10, 50, 5):
        data_path = './Inputs/P=%d' % i
        (objVal, cost) = heuristic_delta_weight(data_path, converge_count=2, tolerance=2, d1=100, d2=0)
        d.loc[d_idx] = [i, objVal, cost]
        d_idx += 1
    d.to_csv('heuristic_MIP_model.csv', index=False)
