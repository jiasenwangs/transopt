# -*- coding: utf-8 -*-
"""
@author: Jiasen Wang
The code includes testing functions
"""

def obj1(x,parameter,np):
    a1=parameter[0]
    f = a1*np.cos(x[0])*np.sin(x[1])-x[0]/(1+x[1]**2)
    return f

# put parameter and np as input, to ensure the function structure for unified call format
def h1(x,parameter,np):
    ceq = []
    return ceq

#lb = [-1, -1]
#ub = [2, 1]
def g1(x,parameter,np):
    c = [-1-x[0],x[0]-2,-1-x[1],x[1]-1]
    return c

def obj1_optimal(parameter,np,obj1):
    assert parameter[0]==1.0
    x=[2,0.10578]
    f = obj1(x,parameter,np)
    return x,f

def obj2(x,parameter,np):
    a1 = parameter[0]
    f = a1*(x[0]-1)**2+(x[0]-x[1])**2+(x[1]-x[2])**3+(x[2]-x[3])**4+(x[3]-x[4])**4
    return f

#n_eq = 3, n_ineq = 10
def h2(x,parameter,np):
    ceq = [x[0]+x[1]**2+x[2]**3-3*np.sqrt(2)-2,x[1]-x[2]**2+x[3]-2*np.sqrt(2)+2,x[0]*x[4]-2]
    return ceq

def g2(x,parameter,np):
    c = [-5-x[0],x[0]-5,-5-x[1],x[1]-5,-5-x[2],x[2]-5,-5-x[3],x[3]-5,-5-x[4],x[4]-5]
    return c

def obj2_optimal(parameter,np,obj2):
    assert 1.0==parameter[0]
    x=[1.1166,1.2204,1.5378,1.9728,1.7911]
    f = obj2(x,parameter,np)
    return x,f

def obj3(x,parameter,np):
    a1 = parameter[0]
    f = -a1*x[3]
    return f
#n_eq = 4, n_ineq = 
def h3(x,parameter,np):
    k1 = 0.09755988
    k2 = 0.99*k1
    k3 = 0.0391908
    k4 = 0.9*k3
    ceq = [x[0]+k1*x[0]*x[4]-1,x[1]-x[0]+k2*x[1]*x[5],x[2]+x[0]+k3*x[2]*x[4]-1,x[3]-x[2]+x[1]-x[0]+k4*x[3]*x[5]]
    return ceq

# lb = [0,0,0,0,0,0]
# ub = [1,1,1,1,16,16]
def g3(x,parameter,np):
    c = [x[4]**0.5+x[5]**0.5-4,-x[0],x[0]-1,-x[1],x[1]-1,-x[2],x[2]-1,-x[3],x[3]-1,-x[4],x[4]-16,-x[5],x[5]-16]
    return c

def obj3_optimal(parameter,np,obj3):
    assert 1.0==parameter[0]
    x=[0.772,0.517,0.204,0.388,3.036,5.097]
    f = obj3(x,parameter,np)
    return x,f

def obj4(x,parameter,np):
    a1 = parameter[0]
    n=5
    s1 = [x[i]**2 for i in range(n)]
    s1 = np.sqrt(sum(s1)/n)
    s2 = [np.cos(2*np.pi*x[i]) for i in range(n)]
    s2 = sum(s2)/n
    f=-20.0*a1*np.exp(-0.2*s1)-np.exp(s2)+20+np.exp(1)
    return f

def h4(x,parameter,np):
    ceq = []
    return ceq

# lb = [0,0,0,0,0,0]
# ub = [1,1,1,1,16,16]
def g4(x,parameter,np):
    c=[]
    n=5
    for i in range(n):
        c.extend((-x[i],x[i]-2.5))
    return c

def obj4_optimal(parameter,np,obj4):
    assert 1.0==parameter[0]
    n=5
    x=[0.0 for i in range(n)]
    f = obj4(x,parameter,np)
    return x,f

def obj5(x,parameter,np):
    a1 = parameter[0]
    x1 = x[0]
    x2 = x[1]
    y1 = x[2]
    y2 = x[3]
    y3 = x[4]
    f=a1*2*x1+3*x2+1.5*y1+2*y2-0.5*y3
    return f

def h5(x,parameter,np):
    x1 = x[0]
    x2 = x[1]
    y1 = x[2]
    y2 = x[3]
    ceq = [x1**2+y1-1.25,x2**1.5+1.5*y2-3]
    return ceq

def g5(x,parameter,np):
    x1 = x[0]
    x2 = x[1]
    y1 = x[2]
    y2 = x[3]
    y3 = x[4]
    c=[x1+y1-1.6,1.333*x2+y2-3,-y1-y2+y3,-x1,-x2]
    return c

def obj5_optimal(parameter,np,obj5):
    assert 1.0==parameter[0]
    x=[1.12,1.31,0,1,1]
    f = obj5(x,parameter,np)
    return x,f

def obj6(x,parameter,np):
    a1 = parameter[0]
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
    Q=np.matrix(Q)
    x=np.matrix([x])
    f=a1*x*Q*x.T
    f=f[0,0]
    return f

def h6(x,parameter,np):
    ceq = []
    return ceq

def g6(x,parameter,np):
    c=[]
    return c

def obj6_optimal(parameter,np,obj6):
    assert 1.0==parameter[0]
    x=[1,1,0,0,1,0,1,1,0,0]
    f = obj6(x,parameter,np)
    return x,f

def obj7(x,parameter,np):
    a1=parameter[0]
    f = a1*x[0]**4-3*x[0]**3-1.5*x[0]**2+10*x[0]
    return f

# put parameter and np as input, to ensure the function structure for unified call format
def h7(x,parameter,np):
    ceq = []
    return ceq

def g7(x,parameter,np):
    c = [-5-x[0],x[0]-5]
    return c

def obj7_optimal(parameter,np,obj7):
    assert parameter[0]==1.0
    x=[-1]
    f = obj7(x,parameter,np)
    return x,f

def obj8(x,parameter,np):
    a1 = parameter[0]
    n=100
    s1 = [x[i]**2 for i in range(n)]
    s1 = np.sqrt(sum(s1)/n)
    s2 = [np.cos(2*np.pi*x[i]) for i in range(n)]
    s2 = sum(s2)/n
    f=-20.0*a1*np.exp(-0.2*s1)-np.exp(s2)+20+np.exp(1)
    return f

def h8(x,parameter,np):
    ceq = []
    return ceq

def g8(x,parameter,np):
    c=[]
    n=100
    for i in range(n):
        c.extend((-x[i],x[i]-2.5))
    return c

def obj8_optimal(parameter,np,obj8):
    assert 1.0==parameter[0]
    n=100
    x=[0.0 for i in range(n)]
    f = obj8(x,parameter,np)
    return x,f

## for using of scipy.minimize
## problem 1: n_eq=0, n_ineq=4 # may write as bound constraint, special='p1'
## problem 2: n_eq = 3, n_ineq = 10, special='p2'
## problem 3: n_eq = 4, n_ineq = 13, special='p3'
## problem 4: n_eq = 0, n_ineq = 10, special='p4'
## problem 5: n_eq = 2, n_ineq = 5, special=='p5'
## problem 6: n_eq = 0, n_ineq = 0, special=='p6'
## problem 7: n_eq = 0, n_ineq = 2, special='p7'

def cons_basic(parameter,np,h,g,partial,special=None):
    #equality
    def h_cons(x,h,parameter,np,idx):
        return h(x,parameter,np)[idx]
    def h_cons_p5(x,idx):
        return x[2+idx]*(x[2+idx]-1)
    def h_cons_p6(x,idx):
        return x[idx]*(x[idx]-1)
    #equality
    def g_cons(x,g,parameter,np,idx):
        return -g(x,parameter,np)[idx]
    #bounds
    def b1():
        bnds = ((-1.0, 2.0) , (-1.0, 1.0)) #-1<=x1<=2.0
        return bnds
    def b2():
        bnds = tuple([(-5.0, 5.0) for i in range(5) ])#-5<=x1<=5
        return bnds
    def b3():
        bnds = tuple([(0.0, 1.0) for i in range(4) ]+[(0.0, 16.0) , (0.0, 16.0)])#-5<=x1<=5
        return bnds
    def b4():
        bnds = tuple([(0.0, 2.5) for i in range(5) ] )#-5<=x1<=5
        return bnds
    def b5():
        bnds = ((0, None) , (0, None), (0, 1), (0, 1), (0, 1))
        return bnds
    def b6():
        bnds = tuple([(0.0, 1.0) for i in range(10) ])
        return bnds
    def b7():
        bnds = ((-5.0, 5.0), ) #-1<=x1<=2.0
        return bnds
    def b8():
        bnds = tuple([(0.0, 2.5) for i in range(100) ] )
        return bnds
    bnds = None
    if special=='p1':
        n_eq=0
        n_ineq=0
        bnds = b1()
    elif special=='p2':
        n_eq=3
        n_ineq=0
        bnds = b2()
    elif special=='p3':
        n_eq=4
        n_ineq=1
        bnds = b3()
    elif special=='p4':
        n_eq=0
        n_ineq=0
        bnds = b4()
    elif special=='p5':
        n_eq=2
        n_ineq=3
        bnds = b5()
    elif special=='p6':
        n_eq=0
        n_ineq=0
        bnds = b6()
    elif special=='p7':
        n_eq=0
        n_ineq=0
        bnds = b7()
    elif special=='p8':
        n_eq=0
        n_ineq=0
        bnds = b8()
    
    cons=[]
    for i in range(n_eq):
        cons.append({'type': 'eq', 'fun': partial(h_cons,h=h,parameter=parameter,np=np,idx=i)})
    if special=='p5':
        for i in range(3):
            cons.append({'type': 'eq', 'fun': partial(h_cons_p5,idx=i)})
    elif special=='p6':
        for i in range(10):
            cons.append({'type': 'eq', 'fun': partial(h_cons_p6,idx=i)})
    
    for i in range(n_ineq):
        cons.append({'type': 'ineq', 'fun': partial(g_cons,g=g,parameter=parameter,np=np,idx=i)})
    
    return cons,bnds

#bnds = tuple([(0.0, 1.0) for xx in x])

if __name__ =="__main__":
    import numpy as np
    # print (int(np.round(0.3)) )
    # raise
    # x=[2,0.10578]
    # print (obj1([1.99999,0.10578],[1],np) )
    # print (obj1([2,0.0],[1],np) )
    # print (0.02/2)
    # test constraint
    # print (g_1([-1,-1]) )
    # print (obj2_optimal([1,1,1,1,1],np,obj2) )
    # print (len( h2([1.1166,1.2204,1.5378,1.9728,1.7911],[1],np)) )
    # print (len(g2([1.1166,1.2204,1.5378,1.9728,1.7911],[1],np)) )
    #[-3.0736967285172057e-05, -5.5964746190628745e-05, -5.7740000000139347e-05]
    # print (obj3_optimal([1.0],np,obj3) )
    # print ( len(h3([0.772,0.517,0.204,0.388,3.036,5.097],[1.0],np) ) ) #TODO-0.001
    # print ( len( g3([0.772,0.517,0.204,0.388,3.036,5.097],[1.0],np) ) )
    # print (obj4_optimal([1.0,1.0,1.0],np,obj4) )
    # print ( len (h4([0 for i in range(5)],[1.0],np) ) )
    # print ( len( g4([2.5 for i in range(5)],[1.0],np) ) )
    # zz =cons_vio([0.772,0.517,0.204,0.388,3.036,5.097],h3,g3,[1.0],np)
    # print (zz)
    # print (obj5_optimal([1.0],np,obj5) )
    # print ( len (h5([1.12,1.31,0,1,1],[1.0],np) ) )
    # print ( len( g5([1.12,1.31,0,1,1],[1.0],np) ) )
    # zz = 2*(2**10-1)
    # print (zz)
    # obj7([1,1,0,0,1,0,1,1,0,0],[1.0],np)
    # print (obj6_optimal([1],np,obj6) )
    # print (obj7([-1],[1],np) )
    # print (obj7_optimal([1],np,obj7))
    # pass
    
    from scipy.optimize import minimize
    import numpy as np
    from functools import partial
    parameter = [1.0]
    fun = lambda x: obj1(x,parameter,np)
    cons,bnds = cons_basic(parameter,np,h1,g1,partial,special='p1')
    x0 = np.array([0.5,0.1])

    fun = lambda x: obj2(x,parameter,np)
    cons,bnds = cons_basic(parameter,np,h2,g2,partial,special='p2')
    # print(bnds)
    x0 = np.array([1,1,1,1,1]) #[1.1166,1.2204,1.5378,1.9728,1.7911]

    fun = lambda x: obj3(x,parameter,np)
    cons,bnds = cons_basic(parameter,np,h3,g3,partial,special='p3')
    # print(bnds)
    x0 = np.array([1,1,1,1,1,1]) #[0.772,0.517,0.204,0.388,3.036,5.097]

    fun = lambda x: obj4(x,parameter,np)
    cons,bnds = cons_basic(parameter,np,h4,g4,partial,special='p4')
    # print(bnds)
    x0 = np.array([0.1 for i in range(5)]) #[0,...,0]

    fun = lambda x: obj5(x,parameter,np)
    cons,bnds = cons_basic(parameter,np,h5,g5,partial,special='p5')
    # print(bnds)
    x0 = np.array([-100,-10,0.1,0.7,0.5]) #[1.12,1.31,0,1,1]

    # fun = lambda x: obj6(x,parameter,np)
    # cons,bnds = cons_basic(parameter,np,h6,g6,partial,special='p6')
    # # print(bnds)
    # x0 = np.array([0,1,0,0,1,0,1,1,0,0]) #[1,1,0,0,1,0,1,1,0,0

    # fun = lambda x: obj7(x,parameter,np)
    # cons,bnds = cons_basic(parameter,np,h7,g7,partial,special='p7')
    # # print(bnds)
    # x0 = np.array([-0.5]) #[-1]

    # fun = lambda x: obj8(x,parameter,np)
    # cons,bnds = cons_basic(parameter,np,h8,g8,partial,special='p8')
    # print(bnds)
    # x0 = np.array([0.1 for i in range(100)]) #[-1]

    
    res = minimize(fun, x0, method='SLSQP', constraints=cons,bounds= bnds)
    print(res.fun) # -0.773684210526435
    print(res.x) # [0.9 0.9 0.1]
    print (res)
    raise