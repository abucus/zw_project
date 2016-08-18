from sys import exit

from gurobipy import Model, GRB, quicksum, LinExpr
from input_data import load_data
from os.path import exists, join
from os import makedirs
from random import choice
from time import clock
from multiprocessing.pool import Pool
from copy import deepcopy
from numpy import argmax, argmin


class HeuristicParallelModel:
    def __init__(self, input_path, output_dir, debug_mode=False):

        if not exists(output_dir):
            makedirs(output_dir)

        self.output_dir = output_dir

        self.pool = Pool()

        self.obj_value_trace = []

        D = load_data(input_path)._asdict()
        for k in D:
            setattr(self, k, D[k])

        self.debug_mode = debug_mode

    def optimize(self):
        start = clock()
        x, q, objVal = self.__anneal()
        cost = clock() - start
        # x, q, objVal = anneal(init_x())
        with open(join(self.output_dir, 'heuristic_x_q.txt'), 'w') as f:
            f.write('Obj Value:\n%r' % objVal)
            f.write('\nx:\n%r' % x)
            f.write('\nq:\n%r' % q)

        return objVal, cost

    def __anneal(self):
        from random import random
        x = self.__init_x()
        q = self.__eval_q(x)
        old_cost, j0 = self.__objective_function(x, q)
        T = 1.0
        T_min = 0.01
        alpha = 0.8
        while T > T_min:
            i = 1
            while i <= 20:
                # stop until find a feasible solution
                while True:
                    x_new, p_changed = self.__neighbor(x, j0)
                    q_new = self.__eval_q(x_new)
                    if self.__constraint_valid(x_new, q_new): break

                new_cost, j0 = self.__objective_function(x_new, q_new, p_changed)

                if hasattr(self, j0) and self.j0 == j0:
                    return x, q, old_cost

                ap = self.__acceptance_probability(old_cost, new_cost, T)
                if ap > random():
                    x = x_new
                    q = q_new
                    old_cost = new_cost
                i += 1
            T = T * alpha
            print('Current temperature', T)
        return x, q, old_cost

    def __init_x(self):
        x = {}

        resources = list(set([r for (r, p) in self.resource_project_demand]))
        resources.sort()

        suppliers = list(set([s for (r, s) in self.resource_supplier_capacity]))
        suppliers.sort()
        # print(resources)

        for r in resources:
            # demands of resource r for different projects order by demand from high to low
            demands = []
            for p in self.project_list:
                if self.resource_project_demand[r, p] > 0:
                    demands.append((p, self.resource_project_demand[r, p]))

            demands.sort(key=lambda x: x[1], reverse=True)
            # print('-' * 50)
            # print('allocate resource %s' % r)
            # print('demands:\n%r' % demands)
            # supply of resource r from different suppliers
            supply = [(s, self.resource_supplier_capacity[r_, s]) for (r_, s) in self.resource_supplier_capacity if
                      r_ == r]
            # print('supply:\n%r' % supply)

            for (p, demand) in demands:
                # print('trying to find %s for %s (%r)' % (r, p, demand))
                # print(len(supply), 'all:%r' % supply)
                supplier_candidates = [s for (s, capacity) in supply if
                                       capacity >= demand]

                if not supplier_candidates:
                    # print('Infeasible')
                    exit()
                supplier = choice(supplier_candidates)
                self.resource_supplier_capacity[r, supplier] -= demand

                x[r, supplier, p] = 1
                # print('project {p} choose supplier {s} (capacity {c}) to provide {r} (demand {d})'.format(
                #     p=p, s=supplier, r=r, c=self.resource_supplier_capacity[r, supplier], d=demand))
        return x

    def __eval_q(self, x):
        q = {}
        for (i, j, k) in x:
            # i resource, j supplier, k project
            q[i, j, k] = 0 if x[i, j, k] == 0 else self.resource_project_demand[i, k]
        return q

    def __objective_function(self, x, q, p_changed=None):
        m = Model("Overall_Model")

        if p_changed is not None:
            j0 = self.project_list.index(p_changed)
            CT = self.last_CT
            CT[j0] = self.optmize_single_project(x, j0, self.project_list, self.project_activity,
                                                 self.resource_supplier_list,
                                                 self.resource_supplier_release_time, self.supplier_project_shipping,
                                                 self.M,
                                                 self.output_dir)
        else:
            CT = {}
            CT_ASYNC = dict()
            for j in range(self.project_n):
                ## solve individual model get Project complete date
                CT_ASYNC[j] = self.pool.apply_async(HeuristicParallelModel.optmize_single_project,
                                                    (x, j, self.project_list, self.project_activity,
                                                     self.resource_supplier_list,
                                                     self.resource_supplier_release_time,
                                                     self.supplier_project_shipping,
                                                     self.M,
                                                     self.output_dir))
            for j in range(self.project_n):
                CT[j] = CT_ASYNC[j].get()

        self.last_CT = deepcopy(CT)
        DT = {}
        TD = {}
        for j in range(self.project_n):
            ## Project Tadeness,construction completion time
            DT[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT%d)" % j)
            TD[j] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(TD%d)" % j)

        DT[-1] = m.addVar(obj=0, vtype=GRB.CONTINUOUS, name="(DT-1)")

        ## Review Sequence z_ij
        z = {}
        for i in range(self.project_n):
            for j in range(self.project_n):
                if i != j:
                    z[i, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (i, j))

        for j in range(self.project_n):
            z[-1, j] = m.addVar(obj=0, vtype=GRB.BINARY, name="(z%d,%d)" % (-1, j))
        m.update();

        #### Add Constraint ####
        ## Constrain 2: project complete data>due data ##
        for j in range(self.project_n):
            m.addConstr(DT[j] - TD[j], GRB.LESS_EQUAL, self.DD[j], name="constraint_2_project_%d" % j)

        ## Constraint 13
        for j in range(self.project_n):
            m.addConstr(DT[j], GRB.GREATER_EQUAL, CT[j] + self.review_duration[j], name="constraint_13_project_%d" % j)

        ## Constraint 14
        for i in range(-1, self.project_n):
            for j in range(self.project_n):
                if i != j:
                    m.addConstr(DT[j], GRB.GREATER_EQUAL, DT[i] - self.M * (1 - z[i, j]) + self.review_duration[j],
                                name="constraint_14_project_%d_project_%d" % (i, j))

        ## Constrain 15
        for j in range(self.project_n):
            m.addConstr(quicksum(z[i, j] for i in range(-1, self.project_n) if i != j), GRB.EQUAL, 1,
                        name="constraint_15_project_%d" % j)

        ## Constrain 16
        m.addConstr(quicksum(z[-1, j] for j in range(self.project_n)), GRB.EQUAL, 1, name="constraint_16")

        ## Constrain 17
        for i in range(self.project_n):
            m.addConstr(quicksum(z[i, j] for j in range(self.project_n) if j != i), GRB.LESS_EQUAL, 1,
                        name="constraint_17_project_%d" % i)
        m.update()

        # Set optimization objective - minimize sum of
        expr = LinExpr()
        for j in range(self.project_n):
            expr.add(self.w[j] * TD[j])
        m.setObjective(expr, GRB.MINIMIZE)
        m.update()

        m.params.presolve = 1
        m.update()

        m.optimize()
        m.write(join(self.output_dir, "heuristic_whole.lp"))
        m.write(join(self.output_dir, "heuristic_whole.sol"))

        self.obj_value_trace.append(m.objVal)
        return m.objVal, argmax([self.w[j] * TD[j].X for j in range(self.project_n)])

    def __neighbor(self, x, j0):
        p0 = self.project_list[j0]
        all_rsp = list(filter(lambda rsp: rsp[2] == p0, list(x.keys())))
        for r, s, p in all_rsp:
            demand = self.resource_project_demand[r, p]
            # get all candidate suppliers with (release+ship time) and enough demand
            supplier_candidates = [
                (s_, self.resource_supplier_release_time[r, s_] + self.supplier_project_shipping[r, s_, p]) for (r_, s_)
                in self.resource_supplier_capacity if
                r_ == r and s_ != s and self.resource_supplier_capacity[r_, s] >= demand]
            supplier_candidates = list(filter(
                lambda x: x[1] < self.resource_supplier_release_time[r, s] + self.supplier_project_shipping[r, s, p],
                supplier_candidates))

            if supplier_candidates:
                x.pop((r, s, p))
                self.resource_supplier_capacity[r, s] += demand

                new_supplier = supplier_candidates[argmin([e[1] for e in supplier_candidates])][0]
                x[r, new_supplier, p] = 1
                self.resource_supplier_capacity[r, new_supplier] -= demand
        return x, p

    @staticmethod
    def optmize_single_project(x, j, project_list, project_activity, resource_supplier_list,
                               resource_supplier_release_time, supplier_project_shipping, M, output_dir):
        '''
        Given the generated x for single project, try to optimize the tardiness of the project.
        '''
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
            # print(project_activities.node[row]['duration'])
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
        m.write(join(output_dir, "heuristic_%d.lp" % j))
        m.write(join(output_dir, "heuristic_%d.sol" % j))
        return m.objVal

    def __constraint_valid(self, x, q):
        '''
        Check whether constraint valid for solution x
        :param x:
        :return: True if x is valid for constraint False else
        '''

        # Constraint 7
        resource_cost = 0
        for (i, j, k) in q:
            resource_cost += self.c[i, j, k] * q[i, j, k]
        return resource_cost <= self.B

    def __acceptance_probability(self, old_cost, new_cost, T):
        from math import e
        return e ** ((old_cost - new_cost) / T)


if __name__ == '__main__':
    ## Generate x init
    # m = HeuristicParallelModel('C:/Users/mteng/Desktop/small case/', 'C:/Users/mteng/Desktop/Heuristic')
    import matplotlib.pyplot as plt

    m = HeuristicParallelModel('./Inputs/P=20/', 'C:/Users/mteng/Desktop/Heuristic')
    (objVal, cost) = m.optimize()

    plt.plot(range(len(m.obj_value_trace)), m.obj_value_trace)
    max_y = max(m.obj_value_trace)
    min_y = min(m.obj_value_trace)
    plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
    plt.show()
    print('ObjVal', objVal, 'cost', cost)
