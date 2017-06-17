from gurobipy import *

try:

    # Create a new model
    m = Model("mip1")

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    m.update()
    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    exp0 = x + 2 * y + 3 * z
    m.addConstr(exp0 <= 4, "c0")

    # Add constraint: x + y >= 1
    exp = x + y
    print(type(exp))
    m.addConstr(exp >= 1, "c1")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

    for c in m.getConstrs():
        print('const %s %g' % (c.ConstrName, c.RHS))

    print(exp0.getValue())
    print(exp.getValue())
except GurobiError as e:
    # print('Error code ' + str(e.errno) + ": " + str(e))
    raise
except AttributeError:
    # print('Encountered an attribute error')
    raise
