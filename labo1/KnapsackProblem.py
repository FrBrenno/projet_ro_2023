"""
The Knapsack Problem for Pulp Modeller
"""

from pulp import *

#Knapsack Model

#Indice
Products = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
#Constants
MaxCapacity = 20
weight = {"A": 2, "B": 5, "C": 7,  "D": 13, "E": 1, "F": 4, "G": 9, "H": 2, "I": 7, "J": 6}
price = { "A": 4, "B": 10, "C": 15, "D": 13, "E": 2, "F": 8, "G": 18, "H": 1, "I": 5, "J": 9}
#Problem
prob = LpProblem("Knapsack Problem", LpMaximize)
vars = LpVariable.dicts("Quantity", Products, 0, None, LpInteger)
#Economic Function
prob += (
    lpSum([vars[p] * price[p] for p in Products]),
    "Sum_of_Knapsack_Value",
)
#Constraint
prob += (
    lpSum([vars[p] * weight[p] for p in Products]) <= MaxCapacity,
    "Sum_of_Knapsack_Weight",
)

#Solving
prob.writeLP("KnapsackProblem.lp")
prob.solve(solver=PULP_CBC_CMD())
print("Status: ", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Total Value of the backpack = ", value(prob.objective))