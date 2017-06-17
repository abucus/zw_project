import time
from multiprocessing.pool import Pool
from os import makedirs
from os.path import exists
from os.path import join

import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr

from input_data import load_data

# logging.basicConfig(filename='my.log', level=_logger.info)

_tardiness_obj_trace = []
_gap_trace = []
_last_x = None
_last_CT = None
_pool = None
_delta_trace = []
_CT_map = {}
_historical_delta_weight_idx_map = {}
_result_output_path = './heuristic/'
_time_limit_per_model = 3600.0

_round = 0
_weight_dataset = pd.DataFrame(columns=['round', 'j', 'k', 'weight'])
_pr_dataset = pd.DataFrame(columns=['round', 'j', 'pr'])
_pa_dataset = pd.DataFrame(columns=['round', 'j', 'k', 'a', 'pa'])
_pa_max_dataset = pd.DataFrame(columns=['round', 'j', 'k', 'max_pa'])
_single_project_objective_dataset = pd.DataFrame(columns=['round', 'project', 'objective value'])
_tardiness_objective_dataset = pd.DataFrame(columns=['round', 'objective value'])

_logger = None


def _time_cal(str, _time_call_trace=[]):
    now = time.clock()
    if _time_call_trace:
        # _logger.info("%s cost %.2f" % (str, now - _time_call_trace[-1]))
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


def _exit_if_infeasible(m):
    if m.status == GRB.INFEASIBLE:
        ilp_file_name = '%s_irreducible_inconsistent_subsystem.ilp' % m.ModelName
        lp_file_name = '%s_LP_model.lp' % m.ModelName
        iis_constraints_file_name = '%s_iis_constraint.txt' % m.ModelName
        print('model %s is infeasible, please see files %s, %s' % (m.ModelName, lp_file_name, ilp_file_name))
        m.computeIIS()
        m.write(ilp_file_name)
        m.write(lp_file_name)

        import pandas as pd
        constraints = pd.DataFrame(columns=['name', 'linear IIS', 'quadratic IIS',
                                            'general IIS', 'sos IIS'])
        variables = pd.DataFrame(columns=['name', 'lower bound IIS', 'upper bound IIS'])

        for c in m.getConstrs():
            if c.IISCONSTR == 1:
                # constraints.loc[constraints.shape[0]] = [c.ConstrName, c.IISCONSTR,
                #                                          getattr(c, 'IISQCONSTR', 0), getattr(c, 'IISGenConstr', 0),
                #                                          getattr(c, 'IISSOS', 0)]
                print(c.ConstrName, 'conflict')

        for v in m.getVars():
            if v.IISLB == 1 or v.IISUB == 1:
                variables.loc[variables.shape[0]] = [v.VarName, v.IISLB, v.IISUB]
                print('variable', v.VarName, 'inconsistent')

        constraints.to_csv('./%s_iis_constraints.csv' % m.ModelName, index=False)
        variables.to_csv('./%s_iis_variables.csv' % m.ModelName, index=False)
        exit(0)


def _normalize(d):
    total = sum(d.values())
    for k, v in d.items():
        d[k] = 1. * v / total


def _objective_function_for_delta_weight(D, delta_weight, d1, d2):
    global _time_limit_per_model, _round, _pr_dataset, _tardiness_objective_dataset
    m = Model("model_for_supplier_assignment")
    m.setParam('OutputFlag', False)
    m.params.timelimit = _time_limit_per_model
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
    # equation 2
    for (r, s) in D.resource_supplier_capacity:
        m.addConstr(quicksum(q[r, s, D.project_list[j]] for j in range(D.project_n)), GRB.LESS_EQUAL,
                    D.resource_supplier_capacity[r, s],
                    name="constraint_3_resource_%s_supplier_%s" % (r, s))

    # constraint 21(4) 23(6)
    for (r, p) in D.resource_project_demand:
        # equation 5
        m.addConstr(quicksum(x[r, i, p] for i in D.resource_supplier_list[r]), GRB.EQUAL, 1,
                    name="constraint_6_resource_%s_project_%s" % (r, p))
        # equation 3
        m.addConstr(quicksum(q[r, i, p] for i in D.resource_supplier_list[r]), GRB.GREATER_EQUAL,
                    D.resource_project_demand[r, p], name="constraint_4_resource_%s_project_%s" % (r, p))

    # constraint 22(5)
    for (i, j, k) in q:
        # i resource, j supplier, k project
        # equation 4
        m.addConstr(q[i, j, k], GRB.LESS_EQUAL, D.M * x[i, j, k],
                    name="constraint_5_resource_%s_supplier_%s_project_%s" % (i, j, k))
    # constraint 7
    shipping_cost_expr = LinExpr()
    for (i, j, k) in q:
        shipping_cost_expr.addTerms(D.c[i, j, k], q[i, j, k])
    # equation 6
    m.addConstr(shipping_cost_expr, GRB.LESS_EQUAL, D.B, name="constraint_7")

    # constraint 8
    # equation 26
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
        p = D.project_list[j]
        for r in [r for (r, p_) in D.resource_project_demand.keys() if p_ == p]:
            expr.add(delta_weight[j, r] * AT[j, r])
    m.setObjective(expr, GRB.MINIMIZE)
    m.update()
    ##########################################
    # m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0
    m.optimize()
    _exit_if_infeasible(m)
    m.write(join(_result_output_path, "round_%d_supplier_assign.lp" % _round))
    m.write(join(_result_output_path, "round_%d_supplier_assign.sol" % _round))
    with open(join(log_output_path, 'shipping_cost.txt'), 'a') as fout:
        fout.write('shipping cost: %f\n' % shipping_cost_expr.getValue())
    _logger.info('shipping cost: %f' % shipping_cost_expr.getValue())

    print('status', m.status)
    # m.write(join(_output_path, 'delta_weight.sol'))
    # m.write(join(_output_path, 'delta_weight.lp'))
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
        # print('j', type(j), j)
        # print('r', type(r), r)
        # print('previous weight', type(delta_weight[j, r]), delta_weight[j, r])
        # print('d1', type(d1), d1)
        # print('d2', type(d2), d2)
        # print('sj', type(sj.get(j, 0)), sj.get(j, 0))
        # print('skj', type(skj.get((j, r))), skj.get((j, r)))
        # print('new weight', type(new_delta_weight[j, r]), new_delta_weight[j, r])
        _logger.info(
            'r[%d,%s] = %f *(1+%f*(%f+%f)*%f) = %f' % (
                j, r, delta_weight[j, r], d1, d2, sj.get(j, 0), skj.get((j, r), 0), new_delta_weight[j, r]))

        # new_delta_weight[j, r] = 1
    _normalize(new_delta_weight)

    for j, r in new_delta_weight.keys():
        # _logger.info('j:' + str(j))
        # _logger.info('r:' + str(r))
        # _logger.info(str([_round, j, r, new_delta_weight[j, r]]))
        _weight_dataset.loc[_weight_dataset.shape[0]] = [_round, j, r, new_delta_weight[j, r]]

    for j in range(D.project_n):
        _pr_dataset.loc[_pr_dataset.shape[0]] = [_round, j, sj.get(j, 0)]

    _tardiness_objective_dataset.loc[_tardiness_objective_dataset.shape[0]] = [_round, tardiness_obj_val]

    return new_delta_weight


def _get_y_for_activities(y, a1, a2):
    if (a1, a2) in y:
        return y[a1, a2].X
    else:
        return 0


def _sensitivity_for_constraints(AT, j, project, y_, project_activity, M):
    global _round, _pa_dataset
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
    # equation 20
    for a in project_activities.nodes():
        for r in project_activities.node[a]['resources']:
            m.addConstr(ST[a], GRB.GREATER_EQUAL, AT[j, r],
                        name="constraint_8_project_%d_activity_%s_resource_%s" % (j, a, r))

    ## Constrain 9 activity sequence constrain
    # equation 21
    for row1, row2 in project_activities.edges():
        m.addConstr(ST[row1] + project_activities.node[row1]['duration'], GRB.LESS_EQUAL,
                    ST[row2], name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))

    ## Constrain 10,11
    for row1 in project_activities.nodes():
        for row2 in project_activities.nodes():
            if row1 != row2 and len(list(
                    set(project_activities.node[row1]['rk_resources']).intersection(
                        project_activities.node[row2]['rk_resources']))) > 0:
                # equation 22
                m.addConstr(ST[row1] + project_activities.node[row1]['duration'] - M * (
                    1 - _get_y_for_activities(y_, row1, row2)), GRB.LESS_EQUAL, ST[row2],
                            name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                # equation 23
                m.addConstr(
                    ST[row2] + project_activities.node[row2]['duration'] - M * _get_y_for_activities(y_, row1, row2),
                    GRB.LESS_EQUAL, ST[row1],
                    name="constraint_11_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                # m.addConstr(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)

    ## Constrain 12
    # equation 24
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
    # m.params.presolve = 1
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

            # if c.Pi != 0:
            #     logging.debug('project %d binding resource:%s Pi:%.4g' % (j, splits[-1], c.Pi))
            # else:
            #     logging.debug('project %d not binding resource:%s Pi:%.4g' % (j, splits[-1], c.Pi))

            _pa_dataset.loc[_pa_dataset.shape[0]] = [_round, j, r, splits[5], c.Pi]
    _skj = {}
    for r in _skja:
        _skj[j, r] = max(_skja[r])
        _pa_max_dataset.loc[_pa_max_dataset.shape[0]] = [_round, j, r, max(_skja[r])]
    return _skj


def optimize_single_project(AT, j, project_list, project_activity, M):
    global _time_limit_per_model
    m = Model("SingleProject_%d" % j)
    m.params.timelimit = _time_limit_per_model
    # m.setParam('OutputFlag', False)
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
    # equation 20
    for a in project_activities.nodes():
        for r in project_activities.node[a]['resources']:
            m.addConstr(AT[j, r], GRB.LESS_EQUAL, ST[a],
                        name="constraint_8_project_%d_activity_%s_resource_%s" % (j, a, r))

    ## Constrain 9 activity sequence constrain
    # equation 21
    for row1, row2 in project_activities.edges():
        m.addConstr(ST[row1] + project_activities.node[row1]['duration'], GRB.LESS_EQUAL,
                    ST[row2], name="constraint_9_project_%d_activity_%s_activity_%s" % (j, row1, row2))

    ## Constrain 10,11
    for row1 in project_activities.nodes():
        for row2 in project_activities.nodes():
            if row1 != row2 and len(list(
                    set(project_activities.node[row1]['rk_resources']).intersection(
                        project_activities.node[row2]['rk_resources']))) > 0:
                # equation 22
                m.addConstr(ST[row1] + project_activities.node[row1]['duration'] - M * (
                    1 - y[row1, row2]), GRB.LESS_EQUAL, ST[row2],
                            name="constraint_10_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                # equation 23
                m.addConstr(
                    ST[row2] + project_activities.node[row2]['duration'] - M * (y[row1, row2]),
                    GRB.LESS_EQUAL, ST[row1],
                    name="constraint_11_project_%d_activity_%s_activity_%s" % (j, row1, row2))
                # m.addConstr(y[j,row1,row2]+y[j,row2,row1],GRB.LESS_EQUAL,1)

    ## Constrain 12
    # equation 24
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
    # m.params.presolve = 1
    m.update()
    # Solve
    # m.params.presolve=0

    m.optimize()
    m.write(join(_result_output_path, "round_%d_optimize_single_project_%d.lp" % (_round, j)))
    m.write(join(_result_output_path, "round_%d_optimize_single_project_%d.sol" % (_round, j)))
    # _logger.info("project %d with optimalVal %r" % (j, m.objVal))
    # m.fixedModel()

    skj = _sensitivity_for_constraints(AT, j, project, y, project_activity, M)
    # for c in m.getConstrs():
    #     if c.ConstrName.startswith('constraint_8_project'):
    #         splits = c.ConstrName.split('_')
    #         if c.Pi == 0:
    #             _logger.info('project %d bind resource:%s Slack:%.4g'%(j, splits[-1],c.Pi))
    #             break
    # else:
    #     _logger.info('project %d not bind'%j)
    _single_project_objective_dataset.loc[_single_project_objective_dataset.shape[0]] = [_round, j, m.objVal]
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
    # equation 17
    for j in range(D.project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, D.DD[j], name="constraint_2_project_%d" % j)

    ## Constraint 13
    # equation 12
    for j in range(D.project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + D.review_duration[j],
                    name="constraint_13_project_%d" % j)

    ## Constraint 14
    # equation 13
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

    # m.params.presolve = 1
    m.update()

    m.optimize()

    # _logger.info("mm binding info:")
    sj = {}
    for c in m.getConstrs():
        if c.ConstrName.startswith('constraint_13'):
            j = int(c.ConstrName.split('_')[-1])
            # if c.Pi != 0:
            # sj[j] = 1
            # _logger.info('%s binding Pi:%.4g' % (c.ConstrName, c.Pi))
            # pass
            # else:
            # sj[j] = 0
            # _logger.info('%s not binding Pi:%.4g' % (c.ConstrName, c.Pi))
            # pass
            sj[j] = c.Pi
    return sj


def _objective_function_for_tardiness(x, AT, D):
    global _last_CT, _last_x, _pool, _time_limit_per_model
    m = Model("Overall_Model")
    m.params.timelimit = _time_limit_per_model
    # m.setParam('OutputFlag', False)
    # m.params.IntFeasTol = 1e-7

    CT = {}
    # CT_ASYNC = dict()
    # project_to_recompute, project_no_recompute = get_project_to_recompute(x, D.project_n, D.project_list)
    project_suppliers = _get_project_suppliers_map(x, D.project_list)
    critical_project_resource = dict()
    # for j in range(D.project_n):
    #     p = D.project_list[j]
    #     CT_ASYNC[j] = _pool.apply_async(optimize_single_project,
    #                                     (AT, j, D.project_list, D.project_activity, D.M))
    #
    # for j in CT_ASYNC:
    #     p = D.project_list[j]
    #     CT[j], skj = CT_ASYNC[j].get()
    #     _put_CT(p, project_suppliers[p], CT[j])
    #     _put_historical_delta_weight_idx_map(p, project_suppliers[p], skj)
    #     critical_project_resource.update(skj)

    for j in range(D.project_n):
        p = D.project_list[j]
        CT[j], skj = optimize_single_project(AT, j, D.project_list, D.project_activity, D.M)
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
    # equation 17
    for j in range(D.project_n):
        m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, D.DD[j], name="constraint_2_project_%d" % j)

    ## Constraint 13
    # equation 12
    for j in range(D.project_n):
        m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + D.review_duration[j],
                    name="constraint_13_project_%d" % j)

    ## Constraint 14
    # equation 13
    for i in range(-1, D.project_n):
        for j in range(D.project_n):
            if i != j:
                m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - D.M * (1 - z[i, j]) + D.review_duration[j],
                            name="constraint_14_project_%d_project_%d" % (i, j))

    ## Constrain 15
    # equation 14
    for j in range(D.project_n):
        m.addConstr(quicksum(z[i, j] for i in range(-1, D.project_n) if i != j), GRB.EQUAL, 1,
                    name="constraint_15_project_%d" % j)

    ## Constrain 16
    # equation 15
    m.addConstr(quicksum(z[-1, j] for j in range(D.project_n)), GRB.EQUAL, 1, name="constraint_16")

    ## Constrain 17
    # equation 16
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

    # m.params.presolve = 1
    m.update()

    m.optimize()
    m.write(join(_result_output_path, 'round_%d_tardiness.sol' % _round))
    m.write(join(_result_output_path, 'round_%d_tardiness.lp' % _round))
    z_ = {}
    for i, j in z:
        z_[i, j] = z[i, j].X

    critical_projs = _sensitivity_analysis_for_tardiness(z_, CT, D)

    _tardiness_obj_trace.append(m.objVal)
    _gap_trace.append(m.MIPGap)

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
    from random import seed
    seed(13)

    start = time.clock()
    global _last_x, _last_CT, _pool, _delta_trace, _historical_delta_weight_idx_map, _result_output_path, _time_limit_per_model, _gap_trace, _round
    _tardiness_obj_trace.clear()
    _gap_trace.clear()
    _delta_trace.clear()
    _CT_map.clear()
    _historical_delta_weight_idx_map.clear()
    _last_x = None
    _last_CT = None
    _pool = Pool(2)
    if output_path is not None:
        _result_output_path = output_path
    if not exists(_result_output_path):
        makedirs(_result_output_path)
    D = load_data(input_path)
    _round = 0

    # initialization for GA
    _time_limit_per_model = 3600.0 / (D.project_n + 2)
    delta_weight = {}
    for j in range(D.project_n):
        p = D.project_list[j]
        for r in sorted([r_ for (r_, p_) in D.resource_project_demand.keys() if p_ == p]):
            delta_weight[j, r] = 1  # random()

    # delta_weight[0, 'NK0g2'] = 1
    _logger.info(str(delta_weight))
    _normalize(delta_weight)

    for (j, r) in delta_weight.keys():
        _weight_dataset.loc[_weight_dataset.shape[0]] = [_round, j, r, delta_weight[j, r]]

    optimal = 1e10
    current_converge_count = 0

    with open('trace.log', 'a') as f:
        while current_converge_count < converge_count:
            _round += 1
            _logger.info('-' * 50)
            _logger.info('round %d' % _round)
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
            f.write('%r\n' % _tardiness_obj_trace)
            f.write("time cost:%r" % (time.clock() - start))
            # break
            # print("current_converge_count:", current_converge_count)
            # print("delta size:", len(delta_weight))
            # print(delta_weight)

    return min(_tardiness_obj_trace), time.clock() - start, _gap_trace[np.argmin(_tardiness_obj_trace)]


if __name__ == '__main__':
    # C:/Users/mteng/Desktop/small case/
    # ./Inputs/P=10
    # ./Inputs/case1

    # ### run single
    from log import setup_logger

    # for i in range(6, 7):
    _result_output_path = 'C:/Users/mteng/Desktop/P123/P3/output/result'
    data_path = 'C:/Users/mteng/Desktop/P123/P3'
    log_output_path = 'C:/Users/mteng/Desktop/P123/P3/output/log'

    if not exists(log_output_path):
        makedirs(log_output_path)

    _logger = setup_logger('20170613', '%s/my.log' % log_output_path)
    (objVal, cost, gap) = heuristic_delta_weight(data_path, converge_count=20, tolerance=0.5, d1=10, d2=0.1)
    print(objVal, cost, gap)

    # write intermediate result to file
    _weight_dataset.to_csv('%s/log-weight.csv' % log_output_path, index=False)
    _pa_dataset.to_csv('%s/log-pa.csv' % log_output_path, index=False)
    _pr_dataset.to_csv('%s/log-pr.csv' % log_output_path, index=False)
    _single_project_objective_dataset.to_csv('%s/log-single-project-objective.csv' % log_output_path, index=False)
    _tardiness_objective_dataset.to_csv('%s/log-tardiness-objective.csv' % log_output_path, index=False)
    _pa_max_dataset.to_csv('%s/log-pa-max.csv' % log_output_path, index=False)

    # clear existing dataset
    _weight_dataset.drop(_weight_dataset.index, inplace=True)
    _pa_dataset.drop(_pa_dataset.index, inplace=True)
    _pr_dataset.drop(_pr_dataset.index, inplace=True)
    _single_project_objective_dataset.drop(_single_project_objective_dataset.index, inplace=True)
    _tardiness_objective_dataset.drop(_tardiness_objective_dataset.index, inplace=True)
    _pa_max_dataset.drop(_pa_max_dataset.index, inplace=True)

    # for i in range(5, 10):
    #     # for i in range(6, 7):
    #     data_path = './Inputs/New_New_P=%d/' % i
    #     output_path = 'c:/Users/mteng/Desktop/result/New_New_P=%d' % i
    #
    #     if not exists(output_path):
    #         makedirs(output_path)
    #
    #     _logger = setup_logger('New_P=%d' % i, '%s/my.log' % output_path)
    #     (objVal, cost, gap) = heuristic_delta_weight(data_path, converge_count=20, tolerance=0.5, d1=10, d2=0.1)
    #     print(objVal, cost, gap)
    #
    #     _weight_dataset.to_csv('%s/log-weight.csv' % output_path, index=False)
    #     _pa_dataset.to_csv('%s/log-pa.csv' % output_path, index=False)
    #     _pr_dataset.to_csv('%s/log-pr.csv' % output_path, index=False)
    #     _single_project_objective_dataset.to_csv('%s/log-single-project-objective.csv' % output_path, index=False)
    #     _tardiness_objective_dataset.to_csv('%s/log-tardiness-objective.csv' % output_path, index=False)
    #     _pa_max_dataset.to_csv('%s/log-pa-max.csv' % output_path, index=False)
    #
    #     _weight_dataset.drop(_weight_dataset.index, inplace=True)
    #     _pa_dataset.drop(_pa_dataset.index, inplace=True)
    #     _pr_dataset.drop(_pr_dataset.index, inplace=True)
    #     _single_project_objective_dataset.drop(_single_project_objective_dataset.index, inplace=True)
    #     _tardiness_objective_dataset.drop(_tardiness_objective_dataset.index, inplace=True)
    #     _pa_max_dataset.drop(_pa_max_dataset.index, inplace=True)


## run in batch for project
# import pandas as pd
#
# d = pd.DataFrame(columns=["Project Size", "Objective Value", "Time Cost", "Gap"])
# d_idx = 0
# for i in range(10, 19, 2)[:1]:  # for project num 3,5,7,9,    range(from,to,step)
#     data_path = './Inputs/P=%d' % i
#     (objVal, cost, gap) = heuristic_delta_weight(data_path, converge_count=1, tolerance=2, d1=100, d2=0.1)
#     d.loc[d_idx] = [i, objVal, cost, gap]
#     d_idx += 1
#     d.to_csv('heuristic_MIP_model_project.csv', index=False)

# ## run in batch for activity
# import pandas as pd
# d = pd.DataFrame(columns=["Activity Number", "Objective Value", "Time Cost", "Gap"])
# d_idx = 0
# for i in range(5, 11, 1): # for activity number 5,6,7,8,9   range(from,to,step)
#     data_path = './Inputs/A=%d' % i
#     (objVal, cost, gap) = heuristic_delta_weight(data_path, converge_count=2, tolerance=2, d1=100, d2=0.1)
#     d.loc[d_idx] = [i, objVal, cost, gap]
#     d_idx += 1
#     d.to_csv('heuristic_MIP_model_activity.csv', index=False)

### run in batch for nk resource
# import pandas as pd
# d = pd.DataFrame(columns=["NK-Resource Number", "Objective Value", "Time Cost", "Gap])
# d_idx = 0
# for i in range(5, 30, 5):  # for nk-resource 5,10,15    range(from,to,step)
#     data_path = './Inputs/NKR=%d' % i
#     (objVal, cost, gap) = heuristic_delta_weight(data_path, converge_count=2, tolerance=2, d1=100, d2=0.1)
#     d.loc[d_idx] = [i, objVal, cost, gap]
#     d_idx += 1
#     d.to_csv('heuristic_MIP_model_nk_resource.csv', index=False)
