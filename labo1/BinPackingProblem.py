from pulp import *

#Indice
Obj = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
#Constants
capacityBin = 13
nbMaxBin = 10
weight = {"A": 2, "B": 5, "C": 7,  "D": 13, "E": 1, "F": 4, "G": 9, "H": 2, "I": 7, "J": 6}
nbTotalObj = 10
#Problem
prob = LpProblem("Bin Packing Problem", LpMinimize)
y = LpVariable.dicts("BinUsed", range(nbMaxBin), 0, 1, LpInteger)
possible_ItemInBin = [(obj, b) for obj in Obj for b in range(nbMaxBin)]
x = LpVariable.dicts("itemInBin", possible_ItemInBin, 0, 1, LpInteger)
#Economic Function
prob += (
    lpSum([y[i] for i in range(nbMaxBin)]),
    "Sum_of_Containers_Used",
)
#Constraint
for j in Obj:
    prob += (
        lpSum([x[(j, i)] for i in range(nbMaxBin)]) == 1,
        f"An item can be in only 1 bin: object {j}",
    )

for i in range(nbMaxBin):
    prob += (
        lpSum([weight[Obj[j]]*x[(Obj[j], i)] for j in range(nbTotalObj)]) <= capacityBin*y[i],
        f"The sum of all items must be smaller than the bin capacity: bin {i}",
    )
    
prob.writeLP("BPP.lp")
prob.solve(solver=PULP_CBC_CMD())
print("Status: ", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Total Number of the Containers used = ", value(prob.objective))