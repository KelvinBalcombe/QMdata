dosympy=True


def pbold(term):
    x=''
    x='\033[1;31m' +  term + '\033[0m'
    return  x

def helpme():
    print('In order to find help on a topic you need to input a number. The help output in red can be copied and pasted into a cell and run without alteration as an example')
    n=input('FOR HELP ON:' +
            '\n Changing the size of plots, enter 1'+  
            '\n Plotting functions, enter 2'+  
            '\n Setting equations to zero enter 3'+  
            '\n Specifying Equations, enter 4'+  
            '\n Solving Simultaneous Equations, enter 5'+  
            '\n Defining Functions enter 6' +
            '\n Differentiating, enter 7'   +  
            '\n Indefinite Integration, enter 8'+  
            '\n Finding the critical points of equations and there second order derivatives at these points, enter 9'+   
            '\n Partial derivatives, enter 10'+
            '\n Definite Integrals, enter 11'+
            '\n Matrices, enter 12'+
            '\n'  )
    if n==str(1): 
        print('\nTo make plots larger or smaller use \n'+ 'pltsize(verticalsize,horizontalsize) \n e.g. to give vertical depth of 8 and horzontal length of 12 use:\n' + pbold('pltsize(8,12)'))
    elif n==str(2):
        print('\n To plot a function use plot(somefunction) e.g. to plot the square of x') 
        print(pbold('   plot(x**2)'))
        print('\n If you want to specify the x axis differently you can change by adding the range for the x axis e.g.')
        print(pbold('   plot(x**2,(x,0,5))'))
        print('\n To plot more than one function you can do that also e.g. ') 
        print(pbold('   plot(x**2,x**3)'))
    elif n==str(3):
        print('\n To solve a function use solv1(somefunction) e.g. to solve x**2=1')
        print(pbold('solv1(x**2-1)'))
    elif n==str(4):
        print('\n To specify and equation type eq(y,function of x) e.g. to specify y=x**2 ')
        print(pbold('eq(y,x**2)'))
        print('To give the equation a name e.g e1 you can type')
        print(pbold('e1=eq(y,x**2)'))
    elif n==str(5):
        print('\n To get plots and solve simultaneous equations you can use the plotsimult command from the shortcuts module. However if you want to solve simultaneous equations using the sympy options here is an example of two equations and a solution:')
        print(pbold('e1=eq(y,x); e2=eq(y,2*x+2); solution=solv2(e1,e2)'))
        print('Then have a look at these equations and the solution using\n' +pbold('[e1,e2,solution]'))
    elif n==str(6):
        print('\n To specify a function you need a statement such as def f(x): return somefunctionofx  for example if we want to define the function f(x) that is the square of x we would specify')
        print(pbold('def f(x): return x**2' )) 
        print('If you then input a number it will return the value of the function at that point e.g')
        print(pbold('f(2)'))
        print('Note that if you replace f() with g() or h() then this is perfectly fine') 
        print('Functions can be for more than one argument e.g. the following will square x and and the square of z')
        print(pbold('def f(x,z): return x**2+z**2' ))   
        print('If you then input a numbers it will return the value of the function at that point e.g')
        print(pbold('f(2,3)'))      
              
    elif n==str(7):   
        print('You can differentiate directly by specifying the function of x using diff(some function of x). For example if you wish to differentiate the square of x you can define')
        print(pbold('diff(x**2)'))
        print('Providing the function f() has been defined you can also differentiate the function e.g.')
        print(pbold('def f(x):return x**2; \ndiff(f(x))'))
        print('If you want to get the second order derivative you can specify the order e.g. the second order derivative or x squared with respect to x twice is')
        print(pbold('diff(x**2,x,2)'))
    elif n==str(8):
        print('\n The antiderivative is obtained by using  the integrate command. e.g. To integrate 2*x')
        print(pbold('integrate(2*x)'))
    elif n==str(9):
        print('\n To find the critical points of a function you can use the crits command. For example to find the critical points for the square of x we could use')
        print(pbold('crits(x**2)'))
        print('Providing the function f() has been defined you can also obtain the critical values for that the function e.g.')
        print(pbold('def f(x):return x**2; \ncrits(f(x))'))
    elif n==str(10):
        print('You can partially differentiate directly by specifying the function of x and z using diff(some function of x and z, x or z). For example if you wish to differentiate the square of x+z**2 you can define')
        print(pbold('diff(x+z**2,x)') +' or') 
        print(pbold('diff(x+z**2,z)'))
        print('Providing the function f() has been defined you can also differentiate the function e.g.')
        print(pbold('def f(x,z):return x+ z**2; \n[diff(f(x,z),x),diff(f(x,z),z)] ' ) )  
    elif n==str(11):
        print('\n The definite in is obtained by using  the integrate command specifying the bounds. e.g. To integrate 2*x between 0 and 1 is')
        print(pbold('integrate(2*x,(x,0,1))'))
    elif n==str(12):    
        print('\n Specify both A and B 2 by 2 matrices as follows (each pair [,] is a row)')
        print(pbold('A=M([1,2],[2,3])'))
        print(pbold('B=M([2,2],[3,3])'))
        print('\n Add B and A to get D')
        print(pbold('D=B+A'))
        print('\n Specify a vector b')
        print(pbold('b=M([1,2])'))
        print('\n Transpose the vector b or the matrix A')
        print(pbold('b.T'))
        print(pbold('A.T'))
        print('\nMultiply b times A to get c')
        print(pbold('c=b*A'))
        print('\nMultiply B time A to get D')
        print(pbold('D=B*A'))
        print('\nGet the determinant of the matrix A')
        print(pbold('det(A)'))
        print('\nInvert the matrix A')
        print(pbold('inv(A)'))
        print('\nGet the rank of the matrix A')
        print(pbold('rank(A)'))
        print('\nGet the dimension of the matrix A')
        print(pbold('dim(A)'))
    return

import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from numpy import shape
from sympy import factorial as Factorial
from sympy import summation as sum
from sympy import diff as diff_
from sympy import plot as plot_
from sympy import linear_eq_to_matrix as eqtomatrix

from shortcuts import plotline,plotsimult,plotSD,plotbreakeven,plotfunction,plotlinefrompoints,plotbreakeven_exam

if dosympy:    
    from sympy import *
    
    e=E
    init_printing()
    from sympy.abc import a,b,c,d,q,w,x,y,z,k,m,n,p,r,s,t,A,B,C,D,Q,W,X,Y,Z,K,M,N,P,R,S,T,U
    from sympy.abc import alpha,beta,eta,epsilon,gamma,rho,lamda,mu,omega,phi,sigma,tau,theta
    Sigma,Pi,Omega=symbols('Sigma,Pi,Omega')
    x1,x2,x3,x4,z1,z2,z3,z4,y1,y2,y3,y4=symbols('x1,x2,x3,x4,z1,z2,z3,z4,y1,y2,y3,y4')
    a1,b1,a2,b2,a3,b3,c1,c2,c3,c4,m1,m2,m3,m4,k1,k2,k3,k4,p1,p2,p3,p4=symbols('a1,b1,a2,b2,a3,b3,c1,c2,c3,c4,m1,m2,m3,m4,k1,k2,k3,k4,p1,p2,p3,p4')
    a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44=symbols('a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44')
    b11,b12,b13,b14,b21,b22,b23,b24,b31,b32,b33,b34,b41,b42,b43,b44=symbols('b11,b12,b13,b14,b21,b22,b23,b24,b31,b32,b33,b34,b41,b42,b43,b44') 
    from sympy import Eq as eq
    from sympy.plotting import plot3d  
    
    def log10(x):
        return log(x,10)
    
    def ln(x):
        return log(x,e)
    
    def val(x):
        try:
            return x.evalf()
        except:
            return x
        
    def setzero(inpt):
        #print(type(inpt))
        try:
           inpt=list(inpt)
        except:
            pass
        #print(type(inpt))
        if type (inpt) != list:
           #print('done') 
           inpt=[inpt]
           #print(inpt)
        sol=solve(inpt)
        if type(sol)==dict:
            sol=[sol]
        return sol
    
    def crits(f,x=x):
        if type(f) ==Equality:
            f=f.rhs
        print(['Derivative','Critical Points','2nd Derivative','2nd Derivatives at Critpoints'])
        def funct(): return
        if type(f)==type(funct):
            sol=setzero(diff(f(x),x))
            v=[diff(f(x),x),sol,diff(f(x),x,2)]
            v2=[]
            for i in sol:
                v2=v2+[diff(f(x),x,2).subs(i)]
            return v+[v2]
        else:
            sol=setzero(diff(f,x))
            v=[diff(f,x),sol,diff(f,x,2)]
            v2=[]
            for i in sol:
                v2=v2+[diff(f,x,2).subs(i)]
        return v+[v2]

    def equate(a,b,*args):
        c=a-b
        #print(c)
        return solv1(c,*args)
    
    def solv1(e,*y):
        if y==():
            sy=list(e.free_symbols)[0]
        else:
            sy=y[0]
        s=solve(e,sy)
        t=[]
        for i in s:
            t=t+ [eq(sy,i)]
        return t
    
    def tupletoeq(out,i):
        return eq((tuple(out.items())[i])[0],(tuple(out.items())[i])[1])

    def dicttoeq(out):
        h=[]
        for i in range(len(out)):
            h=h+[tupletoeq(out,i)]
        return h   

    def solv2(e1,e2,*args):
        if type(e1) !=Equality:
            e1=eq(y,e1)
            e2=eq(y,e2)
            #print(e1)
            #print(e2)
        try:    
            return dicttoeq(solve([e1,e2],*args))
        except:
            return (solve([e1,e2],*args))
    
    def ploteq(*args,**kwargs):
        #print(args)
        s=len(args)
        if (type(args[s-1]))==tuple:
            con=True
            cnst=args[s-1]
            s=s-1
        else:
            con=False
            
        if (s==1 and con==False):    
            return plot_(args[0].rhs,**kwargs) 
        if (s==1 and con==True):
            return plot_(args[0].rhs,cnst,**kwargs)
        if s==2 and con==False:
            return plot_(args[0].rhs,args[1].rhs,**kwargs)
        if s==2 and con==True:
            return plot_(args[0].rhs,args[1].rhs,cnst,**kwargs)
        if s==3 and con==False:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,**kwargs)
        if s==3 and con==True:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,cnst,**kwargs)
        if s==4 and con==False:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,args[3].rhs,**kwargs)
        if s==4 and con==True:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,args[3].rhs,cnst,**kwargs)
        if s==5 and con==False:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,args[3].rhs,args[4].rhs,**kwargs)
        if s==5 and con==True:
            return plot_(args[0].rhs,args[1].rhs,args[2].rhs,args[3].rhs,args[4].rhs,cnst,**kwargs)
        
    def plot(*args,**kwargs):
        #use axis_center=(0,0)  to keep 0 being the center 
        if type(args[0]) !=Equality:
            s=len(args)
            return plot_(*args,**kwargs)
        else:
            return ploteq(*args,**kwargs)
    
    def diff(y,*args):
        if type(y) ==Equality:
            return diff_(y.rhs,*args)
        else:
            return diff_(y,*args)
    
    def same(e1,e2):  
        s=simplify(e1-e2)==0
        print(s)
        return eq(e1,e2)

    def makelist(j):
        s=str()
        for i1 in range(j):
            for i2 in range(j):
                s=s+',x'+ str(i1)+str(i2)
        return s        
                
    A=MatrixSymbol('A',2,2)
    B=MatrixSymbol('B',2,2)
    #C=MatrixSymbol('C',2,2)
    
    def rank(A): return A.rank()
    def show(B): return simplify(B)
    
    def inv(A): 
        try:
            return simplify(A**-1)
        except:
            return 'This matrix does not have an inverse'
    
    from sympy.abc import sigma,mu,theta
    
    def M(A,*args):
        s0=shape(A)
        s=shape(args)[0]
        if s0==() and s==0:
           A=[A]
        if s!=0:
            A=[A]
        for i in range(s):
            A=A+[args[i]]
        A=Matrix(A)
        if dim(A)[0]>1 and dim(A)[1]==1:
            A=A.T
        return A  
    
    
    def dim(A):
        return A.shape
    
    
    A44=M([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])  
    B44=M([[b11,b12,b13,b14],[b21,b22,b23,b24],[b31,b32,b33,b34],[b41,b42,b43,b44]])  
    
    def makeA(i,j):
        return A44[0:i,0:j]
    
    def makeB(i,j):
        return B44[0:i,0:j]
    
    def I(i):
        return(M(eye(i)))
    
    def zer0s(i,j=1):
        return(M(zeros(i,j)))
    
def search(k,s=dir()):
    v=list()
    for i in s:
        if k in i:
            v=v+[i]    
    return v    

def frac(x,y):
    return Rational(x,y)

def clearx():
    x=symbols('x')
    return x
def cleary():
    y=symbols('y')
    return y
def clearxy():
    x,y=symbols('x,y')
    return x,y
def clearxyz():
    x,y,z=symbols('x,y,z')
    return x,y,z        

solv1.help='If an expression is put in solv1,it will set that expression to zero and solve for the stated left hand side. If an equation is entered, it will rearange that expression for the stated left hand side'
solv2.help='If 2 expressions are entered, solv2 it will assume these are the rhs of y=expressions and will solve simultaneously.If 2 equations are entered, it will solve them simultaneously'
equate.help='2 expressions need to be entered, these will be set equal and solved'


def pltsize(x1=10,x2=16):
    matplotlib.rcParams['figure.figsize'] = [x2, x1]
    return
