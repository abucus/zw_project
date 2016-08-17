import pandas as pd
from collections import namedtuple


def supplier_assignment_analyze(resource_project_demand, resource_supplier_capacity, x, output_path='./'):
    print('-' * 50)
    print(resource_project_demand)
    print(resource_supplier_capacity)

    resources = sorted([rs[0] for rs in resource_supplier_capacity.keys()])
    projects = sorted([rp[1] for rp in resource_project_demand.keys()])

    FakeVarType = namedtuple('FakeVarType', ['X'])
    fake_var = FakeVarType(-1)
    # setattr(fake_variable, 'X', 0)

    resource_supplier_capacity_dict = {r: list() for r in resources}
    for k, v in resource_supplier_capacity.items():
        resource_supplier_capacity_dict[k[0]].append((k[1], v))
    for v in resource_supplier_capacity_dict.values():
        v.sort(key=lambda e: e[1], reverse=True)
    print(resource_supplier_capacity_dict)

    resource_project_demand_dict = {r: list() for r in resources}
    for k, v in resource_project_demand.items():
        resource_project_demand_dict[k[0]].append((k[1], v))
    for v in resource_project_demand_dict.values():
        v.sort(key=lambda e: e[1], reverse=True)
    print(resource_project_demand_dict)

    for resource in resources:
        supplier_capacities = resource_supplier_capacity_dict[resource]
        supplier_column_lables = ['%s(%d)' % sc for sc in supplier_capacities]
        d = pd.DataFrame(columns=['Project', 'Resource', 'Demand'] + supplier_column_lables)
        d_idx = 0
        pds = resource_project_demand_dict[resource]
        for project_demand in pds:
            d.loc[d_idx] = [project_demand[0], resource, project_demand[1]] \
                           + [x.get((resource, sc[0], project_demand[0]), fake_var).X
                              for sc in supplier_capacities]
            d_idx += 1
        d.to_csv(output_path + 'resource_%s_assignment.csv' % resource, index=False)

def project_resource_suppliery_analyze(model, project_list, w, resource_project_demand, review_duration,
                                       resource_supplier_list, resource_supplier_capacity,
                                       resource_supplier_release_time, c, DD):
    resources = sorted(resource_supplier_list.keys())
    d = pd.DataFrame(columns=['Project', 'w_j_TD_j', 'R_j', 'DD_j']
                             + ['Demand_%s' % r for r in resources]
                             + ['%s_%s_%s' % (r, s, label)
                                for label in ['capacity', 'cost', 'release_time', 'choose_or_not'] for r in resources
                                for s in resource_supplier_list[r]])
    d_idx = 0
    print(resource_project_demand)
    for j, p in enumerate(project_list):
        demands = [resource_project_demand.get((r, p), 0) for r in resources]

        resource_supplier_part = []
        for r in resources:
            for s in resource_supplier_list[r]:
                resource_supplier_part.append(resource_supplier_capacity[r, s])

        for r in resources:
            for s in resource_supplier_list[r]:
                resource_supplier_part.append(c[r, s, p])

        for r in resources:
            for s in resource_supplier_list[r]:
                resource_supplier_part.append(resource_supplier_release_time[r, s])

        for r in resources:
            for s in resource_supplier_list[r]:
                resource_supplier_part.append(model.getVarByName("x_%s_%s_%s" % (r, s, p)).X)

        project_part = [p, w[j] * model.getVarByName("TD_%d" % j).X, review_duration[j], DD[j]]

        d.loc[d_idx] = project_part + demands + resource_supplier_part
        d_idx += 1
    d.to_csv('project_resource_suppliery_analyze.csv', index=False)


def _column_map(resource_supplier_list):
    resources = sorted(resource_supplier_list.keys())
    columns = ['Project', 'w_j_TD_j', 'R_j', 'DD_j'] + ['Demand_%s' % r for r in resources] + [
        '%s_%s_%s' % (r, s, label)
        for label in ['capacity', 'cost', 'release_time', 'choose_or_not'] for r in resources for s in
        resource_supplier_list[r]]

    alphabetics = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    letters = [l for l in alphabetics] + [i + j for i in alphabetics for j in alphabetics]

    for i, c in enumerate(columns):
        print(letters[i], ":", c)


if __name__ == '__main__':
    from input_data import load_data

    D = load_data('Inputs/case1/')
    print(D.resource_supplier_list)
    # project_resource_suppliery_analyze(None, D.project_list, D.w, D.resource_project_demand, D.review_duration,
    #                                    D.resource_supplier_list, D.resource_supplier_capacity,
    #                                    D.resource_supplier_release_time, D.c)
    _column_map(D.resource_supplier_list)
