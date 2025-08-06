# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:43:52 2025

@author: Jiasen Wang
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

def prepareDataset0(pid,parameter = [1]):
    # a1*np.cos(x[0])*np.sin(x[1])-x[0]/(1+x[1]**2)
    # pid = 0
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    # model.setParam('OptimalityTol', 1e-6)
    # model.setParam('FeasibilityTol', 1e-6)
    # parameter = [1]
    a1=parameter[0]

    # Add variables x
    x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=-1, ub=2)  # lb=0 to ensure x >= 0
    x2 = model.addVar(vtype=GRB.CONTINUOUS, name="x2", lb=-1, ub=1)  # lb=0 to ensure y >= 0
    
    # auxilary variables
    x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=-1, ub=1)
    x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4", lb=-1, ub=1)
    x5 = model.addVar(vtype=GRB.CONTINUOUS, name="x5", lb=-1, ub=1)
    x6 = model.addVar(vtype=GRB.CONTINUOUS, name="x6", lb=-1, ub=2)
    x7 = model.addVar(vtype=GRB.CONTINUOUS, name="x7", lb=0, ub=1)
    
    
    model.setObjective(a1*x3-x6, GRB.MINIMIZE)
    

    # Add constraints
    model.addConstr(x3 == x4*x5, "constraint1")
    model.addGenConstrCos(x1, x4,"constraint2")
    model.addGenConstrSin(x2, x5,"constraint3")
    model.addConstr(x6*(1+x7) == x1, "constraint4")
    model.addConstr(x7 == x2*x2, "constraint5")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # optimal_x = [x1.x,x2.x,x3.x,x4.x,x5.x,x6.x,x7.x]
        optimal_x = [x1.x,x2.x]
        optimal_obj_value = model.objVal
        # print(f"Optimal x: {optimal_x}")
        # print(f"Optimal objective value: {optimal_obj_value}")
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset1(pid,parameter = [1]):
    # pid = 1
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    
    a1=parameter[0]

    # Add variables x and y
    x0 = model.addVar(vtype=GRB.CONTINUOUS, name="x0", lb=-5, ub=5)  # lb=0 to ensure x >= 0
    x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=-5, ub=5)  # lb=0 to ensure y >= 0
    x2 = model.addVar(vtype=GRB.CONTINUOUS, name="x2", lb=-5, ub=5)
    x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=-5, ub=5)
    x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4", lb=-5, ub=5)
    
    #(x1-x2)**2 == x5
    x5 = model.addVar(vtype=GRB.CONTINUOUS, name="x5", lb=0, ub=100)
    
    #(x[2]-x[3])**2 == x6
    x6 = model.addVar(vtype=GRB.CONTINUOUS, name="x6", lb=0, ub=100)
    
    #(x[3]-x[4])**2 == x7
    x7 = model.addVar(vtype=GRB.CONTINUOUS, name="x7", lb=0, ub=100)
    
    #x[2]**2 == x8
    x8 = model.addVar(vtype=GRB.CONTINUOUS, name="x8", lb=0, ub=25)
    
    # x = [x0,x1,x2,x3,x4,x5,x6,x7,x8]
    x = [x0,x1,x2,x3,x4]
    
    
    
    #f = a1*(x[0]-1)**2+(x[0]-x[1])**2+(x[1]-x[2])**3+(x[2]-x[3])**4+(x[3]-x[4])**4
    f = a1*(x[0]-1)**2+(x[0]-x[1])**2+(x[1]-x[2])*x5+x6**2+x7**2
    model.setObjective(f, GRB.MINIMIZE)
    #ceq = [x[0]+x[1]**2+x[2]**3-3*np.sqrt(2)-2,x[1]-x[2]**2+x[3]-2*np.sqrt(2)+2,x[0]*x[4]-2]
    #c = [-5-x[0],x[0]-5,-5-x[1],x[1]-5,-5-x[2],x[2]-5,-5-x[3],x[3]-5,-5-x[4],x[4]-5]
    # Add constraints
    model.addConstr(x[0]+x[1]**2+x[2]*x8-3*np.sqrt(2)-2==0, "constraint1")
    model.addConstr(x[1]-x[2]**2+x[3]-2*np.sqrt(2)+2==0, "constraint2")
    model.addConstr(x[0]*x[4]-2==0, "constraint3")
    
    model.addConstr((x1-x2)**2 == x5, "constraint4")
    model.addConstr((x[2]-x[3])**2 == x6, "constraint5")
    model.addConstr((x[3]-x[4])**2 == x7, "constraint6")
    model.addConstr(x[2]**2 == x8, "constraint7")
    

    
    model.optimize()
    # T = 3
    # for i in range(T):
    #     for item in x:
    #         item.start = item.x
    #     model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x]
        optimal_obj_value = model.objVal
        # print(f"Optimal x: {optimal_x}")
        # print(f"Optimal objective value: {optimal_obj_value}")
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset2(pid,parameter = [1]):
    # pid = 2
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    # model.setParam('OptimalityTol', 1e-6)
    # model.setParam('FeasibilityTol', 1e-6)
    # parameter = [1]
    a1=parameter[0]

    # Add variables x and y
    x0 = model.addVar(vtype=GRB.CONTINUOUS, name="x0", lb=0, ub=1)  # lb=0 to ensure x >= 0
    x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=0, ub=1)  # lb=0 to ensure y >= 0
    x2 = model.addVar(vtype=GRB.CONTINUOUS, name="x2", lb=0, ub=1)
    x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=0, ub=1)
    x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4", lb=0, ub=16)
    x5 = model.addVar(vtype=GRB.CONTINUOUS, name="x5", lb=0, ub=16)
    
    x = [x0,x1,x2,x3,x4,x5]
    
    #f = -a1*x[3]
    k1 = 0.09755988
    k2 = 0.99*k1
    k3 = 0.0391908
    k4 = 0.9*k3
    #ceq = [x[0]+k1*x[0]*x[4]-1,x[1]-x[0]+k2*x[1]*x[5],x[2]+x[0]+k3*x[2]*x[4]-1,x[3]-x[2]+x[1]-x[0]+k4*x[3]*x[5]]
    # lb = [0,0,0,0,0,0]
    # ub = [1,1,1,1,16,16]
    #c = [x[4]**0.5+x[5]**0.5-4,-x[0],x[0]-1,-x[1],x[1]-1,-x[2],x[2]-1,-x[3],x[3]-1,-x[4],x[4]-16,-x[5],x[5]-16]
    
    f = -a1*x[3]
    model.setObjective(f, GRB.MINIMIZE)
    # Add constraints
    model.addConstr(x[0]+k1*x[0]*x[4]-1==0, "constraint1")
    model.addConstr(x[1]-x[0]+k2*x[1]*x[5]==0, "constraint2")
    model.addConstr(x[2]+x[0]+k3*x[2]*x[4]-1==0, "constraint3")
    model.addConstr(x[3]-x[2]+x[1]-x[0]+k4*x[3]*x[5] == 0, "constraint4")
    
    model.addConstr(4*x4*x5 <= (16-x4-x5)**2, "constraint5")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x]
        optimal_obj_value = model.objVal
        # print(f"Optimal x: {optimal_x}")
        # print(f"Optimal objective value: {optimal_obj_value}")
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset3(pid,parameter = [1]):
    # pid = 3
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    a1=parameter[0]
    
    n=5
    
    x = [model.addVar(vtype=GRB.CONTINUOUS, name=f"x{i}", lb=0, ub=2.5) for i in range(n)]
    
    # n*s1**2 = gp.quicksum( x[i]**2 for i in range(n) )
    s1 = model.addVar(vtype=GRB.CONTINUOUS, name="s1", lb=0, ub=2.5)
    
    #np.exp(-0.2*s1) = s1_e
    s1_e = model.addVar(vtype=GRB.CONTINUOUS, name="s1_e", lb=np.exp(-0.2*2.5), ub=1)
    # s1_inter = -0.2*s1
    s1_inter = model.addVar(vtype=GRB.CONTINUOUS, name="s1_inter", lb=-0.2*2.5, ub=0)
    
    
    # y[i]==np.cos(2*np.pi*x[i])
    y = [model.addVar(vtype=GRB.CONTINUOUS, name=f"y{i}", lb=-1, ub=1) for i in range(n)]
    
    # 2*np.pi*x[i] = y_inter[i]
    y_inter = [model.addVar(vtype=GRB.CONTINUOUS, name=f"y_inter{i}", lb=0, ub=2*np.pi*2.5) for i in range(n)]
    
    
    #n*s2 = gp.quicksum( y[i] for i in range(n))
    s2 = model.addVar(vtype=GRB.CONTINUOUS, name="s2", lb=-1, ub=1)
    
    #np.exp(s2) = s2_e
    s2_e = model.addVar(vtype=GRB.CONTINUOUS, name="s2_e", lb=np.exp(-1), ub=np.exp(1))

    # s1 = [x[i]**2 for i in range(n)]
    # s1 = np.sqrt(sum(s1)/n)
    # s2 = [np.cos(2*np.pi*x[i]) for i in range(n)]
    # s2 = sum(s2)/n
    # f=-20.0*a1*np.exp(-0.2*s1)-np.exp(s2)+20+np.exp(1)
    f=-20.0*a1*s1_e-s2_e+20+np.exp(1)
    
    model.setObjective(f, GRB.MINIMIZE)
    
    # n=5
    # for i in range(n):
    #     c.extend((-x[i],x[i]-2.5))
    
    # Add constraints
    model.addConstr(n*s1**2 == gp.quicksum( x[i]**2 for i in range(n) ), "constraint1")
    model.addGenConstrExp(s1_inter, s1_e,"constraint2")
    for i in range(n):
        model.addGenConstrCos(y_inter[i],y[i], f"constraint{3+i}")
    
    model.addConstr(n*s2 == gp.quicksum( y[i] for i in range(n)), f"constraint{3+n}")
    model.addGenConstrExp(s2, s2_e, f"constraint{3+n+1}")
    
    model.addConstr(s1_inter == -0.2*s1, f"constraint{3+n+2}")
    
    for i in range(n):
        model.addConstr(2*np.pi*x[i] == y_inter[i], f"constraint{3+n+3+i}")
    model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x]
        optimal_obj_value = model.objVal
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset4(pid,parameter = [1]):
    # pid = 4
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    a1=parameter[0]

    # Add variables x and y
    x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=0)  # lb=0 to ensure x >= 0
    x2 = model.addVar(vtype=GRB.CONTINUOUS, name="x2", lb=0)  # lb=0 to ensure y >= 0
    y1 = model.addVar(vtype=GRB.BINARY, name="y1")
    y2 = model.addVar(vtype=GRB.BINARY, name="y2")
    y3 = model.addVar(vtype=GRB.BINARY, name="y3")
   
    # x3=x2**2
    x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=0)
    
    
    x = [x1,x2]
    y = [y1,y2,y3]
    
    
    f=a1*2*x1+3*x2+1.5*y1+2*y2-0.5*y3
    
    model.setObjective(f, GRB.MINIMIZE)
    # Add constraints
    # ceq = [x1**2+y1-1.25,x2**1.5+1.5*y2-3]
    # c=[x1+y1-1.6,1.333*x2+y2-3,-y1-y2+y3,-x1,-x2]
    model.addConstr(x1**2+y1-1.25==0, "constraint1")
    model.addConstr(x3*x2==(3-1.5*y2)**2, "constraint2")
    model.addConstr(x1+y1-1.6<=0, "constraint3")
    model.addConstr(1.333*x2+y2-3<=0, "constraint4")
    model.addConstr(-y1-y2+y3<=0, "constraint5")
    model.addConstr(x3==x2**2, "constraint6")
    
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x] + [int(item.x) for item in y]
        optimal_obj_value = model.objVal
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset5(pid,parameter = [1]):
    # pid = 5
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    
    a1 = parameter[0]

    # Add variables x and y
    Q=[[-1,-2,2,8,-5,1,-4,0,0,8],
       [-2,2,0,-5,4,-4,-4,-5,0,-5],
       [2,0,2,-3,7,0,-3,7,5,0],
       [8,-5,-3,-1,-3,-1,7,1,7,2],
       [-5,4,7,-3,1,0,-4,2,4,-2],
       [1,-4,0,-1,0,1,9,5,2,0],
       [-4,-4,-3,7,-4,9,3,1,2,0],
       [0,-5,7,1,2,5,1,0,-3,-2],
       [0,0,5,7,4,2,2,-3,2,3],
       [8,-5,0,2,-2,0,0,-2,3,3]]
    n = len(Q)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    
    Q=np.array(Q,float)
    f=a1*gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
    # gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
    
    model.setObjective(f, GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_x = [int(x[i].x) for i in range(n)]
        optimal_obj_value = model.objVal
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def prepareDataset6(pid,parameter = [1]):
    # pid = 6
    
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    
    a1=parameter[0]

    # Add variables x and y
    x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=-5,ub=5)  # lb=0 to ensure x >= 0
    #y1 = x1**2
    y1 = model.addVar(vtype=GRB.CONTINUOUS, name="y1", lb=0,ub=25)
    
    x = [x1]
    # y = [y1,y2,y3]
    
    #f = a1*x[0]**4-3*x[0]**3-1.5*x[0]**2+10*x[0]
    f = a1*y1**2-3*x[0]*y1-1.5*x[0]**2+10*x[0]
    
    model.setObjective(f, GRB.MINIMIZE)
    # c = [-5-x[0],x[0]-5]
    
    model.addConstr(y1 == x1**2, "constraint1")
    model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x]
        optimal_obj_value = model.objVal
    else:
        raise
    return optimal_x

def prepareDataset7(pid,parameter = [1]):
    # pid = 7
    model = gp.Model(f"problem{pid}")
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    # model.setParam('OptimalityTol', 1e-6)
    # model.setParam('FeasibilityTol', 1e-6)
    
    a1=parameter[0]
    
    n=100
    
    x = [model.addVar(vtype=GRB.CONTINUOUS, name=f"x{i}", lb=0, ub=2.5) for i in range(n)]
    
    # n*s1**2 = gp.quicksum( x[i]**2 for i in range(n) )
    s1 = model.addVar(vtype=GRB.CONTINUOUS, name="s1", lb=0, ub=2.5)
    
    #np.exp(-0.2*s1) = s1_e
    s1_e = model.addVar(vtype=GRB.CONTINUOUS, name="s1_e", lb=np.exp(-0.2*2.5), ub=1)
    # s1_inter = -0.2*s1
    s1_inter = model.addVar(vtype=GRB.CONTINUOUS, name="s1_inter", lb=-0.2*2.5, ub=0)
    
    
    # y[i]==np.cos(2*np.pi*x[i])
    y = [model.addVar(vtype=GRB.CONTINUOUS, name=f"y{i}", lb=-1, ub=1) for i in range(n)]
    
    # 2*np.pi*x[i] = y_inter[i]
    y_inter = [model.addVar(vtype=GRB.CONTINUOUS, name=f"y_inter{i}", lb=0, ub=2*np.pi*2.5) for i in range(n)]
    
    
    #n*s2 = gp.quicksum( y[i] for i in range(n))
    s2 = model.addVar(vtype=GRB.CONTINUOUS, name="s2", lb=-1, ub=1)
    
    #np.exp(s2) = s2_e
    s2_e = model.addVar(vtype=GRB.CONTINUOUS, name="s2_e", lb=np.exp(-1), ub=np.exp(1))

    # s1 = [x[i]**2 for i in range(n)]
    # s1 = np.sqrt(sum(s1)/n)
    # s2 = [np.cos(2*np.pi*x[i]) for i in range(n)]
    # s2 = sum(s2)/n
    # f=-20.0*a1*np.exp(-0.2*s1)-np.exp(s2)+20+np.exp(1)
    f=-20.0*a1*s1_e-s2_e+20+np.exp(1)
    
    model.setObjective(f, GRB.MINIMIZE)
    
    # n=5
    # for i in range(n):
    #     c.extend((-x[i],x[i]-2.5))
    
    # Add constraints
    model.addConstr(n*s1**2 == gp.quicksum( x[i]**2 for i in range(n) ), "constraint1")
    model.addGenConstrExp(s1_inter, s1_e,"constraint2")
    for i in range(n):
        model.addGenConstrCos(y_inter[i],y[i], f"constraint{3+i}")
    
    model.addConstr(n*s2 == gp.quicksum( y[i] for i in range(n)), f"constraint{3+n}")
    model.addGenConstrExp(s2, s2_e, f"constraint{3+n+1}")
    
    model.addConstr(s1_inter == -0.2*s1, f"constraint{3+n+2}")
    
    for i in range(n):
        model.addConstr(2*np.pi*x[i] == y_inter[i], f"constraint{3+n+3+i}")
    model.optimize()
    if model.status == GRB.OPTIMAL:
        optimal_x = [item.x for item in x]
        optimal_obj_value = model.objVal
    else:
        # print("No optimal solution found.")
        raise
    return optimal_x

def storeTrainData(istrain=True):
    import pandas as pd
    import csv
    if istrain:# for train
        seed = 0
        sample_num = 10000
    else:# for test
        seed = 666
        sample_num = 100
    rng = np.random.default_rng(seed)
    
    for pid in [2,3,4,5,6,7]:#0,1
    # for pid in range(8):#
        result = []
        print (pid,sample_num)
        func_name= f"prepareDataset{pid}"
        for sample_id in range(sample_num): #10000
            random_integer = rng.integers(0, 2, size=1)[0]
            if random_integer==1: # number > 1
                parameter = list(rng.uniform(1.01, 1.1, 1) )
            else:  # number <1
                parameter = list(rng.uniform(0.9, 0.99, 1) )
            # parameter=[1]
            args = (pid, parameter)
            optimal_x = globals()[func_name](*args)
            
            result.append([pid, str(parameter ), str(optimal_x )])
    
        columns = ['pid', 'parameter', 'optimal_x']
        
        # Save the flattened data to a CSV file
        if istrain:
            csv_filename = f'train_data_problem{pid}.csv'
        else:
            csv_filename = f'valid_data_problem{pid}.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)  # Write the header row
            writer.writerows(result)  # Write the data rows

if __name__ =="__main__":
    # prepareDataset0()
    # prepareDataset1()
    # prepareDataset2()
    # prepareDataset3()
    # prepareDataset4()
    # prepareDataset5()
    # prepareDataset6() 
    # prepareDataset7()
    storeTrainData(True)  # True means saving training data the current dir ./, False means saving validation data