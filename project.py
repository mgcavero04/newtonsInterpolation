import math
import numpy as np
import matplotlib.pyplot as plt
# the lines above are used for the visualization
# This file can be run from command promp(python installed) from its location with: python project2.py

# The program runs calling init(), 
# After initializing some varibles with the node and function values first calls newtons1 function to generate the coefficients and saves then in an array varible
# Then it calls the newtons2 function which generates Pn(x) value and also saves in a variable
# This is done for each of the two x values=0.985 and 0.1
# It also calculates the actual value in function: actualValue(x) for each of the x values.
# Then it prints those calculations

def newtons1(points, f): #Newtons Divided Difference: Returns each of the coefficients x(k) from Newton Interpolation in an array
    returned=f
    n=len(points)
    y =  [[0 for x in range(n)] for y in range(n)] 
    coeficientsMatch=[]
    for i in range(0, len(points)):    
        y[i][0] = returned[i]
    for i in range(1, n): 
        for j in range(n - i):        
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (points[j] - points[i + j]))
    for i in range(0, len(y)):
        coeficientsMatch.append(y[0][i])
    return coeficientsMatch

def newtons2(x, xk, f, nConstants): # Newtons Divided Difference: Returns the Pn value of the interpolation when x (0.985 or 0.1) value is provided 
    pn = f[0]
    for i in range(1, nConstants):
        error = 1; 
        for j in range(i): 
            error = error * (x - xk[j]); 
        pn = pn + (error * f[i])
    return pn

poly1= [] 
poly2= [] 

def f( myList = [], *args ): # Evaluates the function f(x) and returns this values in array
    polyFunction = []
    for x in myList:
        constantX= 1/(1+25*x**2)
        polyFunction.append(constantX)
    return polyFunction

def uniformlyNodes(): # Returns each of 21 Uniform nodes values in an array
    uNodes=[]
    for k in range(0, 21, 1):
        xk=-1+k*0.1
        uNodes.append(xk)
    return uNodes

def chebyshevNodes(): # Returns each of 21 Chebyshe nodes values in an array
    cNodes=[]
    for k in range(0, 21, 1):
        xk=math.cos(((2*k+1)/42)*math.pi)
        cNodes.append(xk)
    return cNodes

def actualValue(x): # Returns the actual function value for x=0.985 or x=0.1
    u=[]
    u.append(x)
    fu=f(u)
    uError=fu[0]
    return uError

def init():
    poly1 = uniformlyNodes() #array of 21 uniform nodes -> x(k): 0,1,...20
    poly2 = chebyshevNodes()
    maxits1=len(poly1) # number of interactions
    maxits2=len(poly2)
    newArray1=f(poly1) # function values array
    newArray2=f(poly2)
    c1=newtons1(poly1, newArray1) # returns array of coefficients from Newtons interpolation with parameters: poly1/poly2=uniform/chebys nodes, newArray1/newArray2=function values
    c2=newtons1(poly2, newArray2)
    xU1=0.985
    xU2=0.1
    aValue1=actualValue(xU1)
    aValue2=actualValue(xU2)
    pnUniform1 = newtons2(xU1, poly1, c1, maxits1) # gets the Newtons Interpolation Polynomial value from parameters: 
    pnUniform2 = newtons2(xU2, poly1, c1, maxits1)
    pnChebys1 = newtons2(xU1, poly2, c2, maxits2)
    pnChebys2 = newtons2(xU2, poly2, c2, maxits2)
    
    print("Final Value of Uniform x=0.985 : " , pnUniform1)
    print("-------------> Uniform Error at x=0.985 : " , abs(pnUniform1 - aValue1))
    print("Final Value of Uniform x=0.1 : " , pnUniform2)
    print("-------------> Uniform Error at x=0.1 : " , abs(pnUniform2 - aValue2))

    print("Final Value of chebyshev x=0.985 : " , pnChebys1)
    print("-------------> chebyshev Error at x=0.985 : " , abs(pnChebys1 -aValue1))
    print("Final Value of chebyshev x=0.1 : " , pnChebys2)
    print("------------> chebyshev Error at x=0.1 : " , abs(pnChebys2 - aValue2))

    # Visualization
    p1=[]
    p2=[]
    polyNew1=np.arange(-1.0+0.985, 1.1+0.985, 0.1) # random values to approximate only for x=0.985 and match the 21 nodes are given here
    for i in polyNew1:
        y1=newtons2(i, poly1, c1, maxits1) # returns an array of f(x) values from interpolation with parameters: i=x value, poly1/poly2=uniform/chebys nodes, c1=coefficients,maxits1=# iterations
        p1.append(y1)
    
    for i in polyNew1:
        y2=newtons2(i, poly2, c2, maxits2)
        p2.append(y2)
        
    plt.subplot(2, 1, 1)
    plt.plot(poly1, newArray1,'ro', label='Actual values') 
    plt.plot(poly1, p1, "-g", label='Pn(x)')
    plt.title("Uniform Polynomial") 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2) 
    plt.plot(poly2, newArray2,'ro', label='Actual values') 
    plt.plot(poly2, p2, "-g", label='Pn(x)')
    plt.title("Chebyshev Polynomial") 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
   
    plt.show()
  
    
init()





