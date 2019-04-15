from typing import List, Callable

class Chebyshev(object): 
    """Object for Chebyshev approximation and related methods 
    
    Note: copied over to python from "Numerical Recipes: 
        the art of scientific computing, 3rd edition"
    
    Args: 
        n: number of total coefficients
        m: number of truncated coefficients
        c: coefficient list
        a: start of approximation interval
        b: end of approximation interval 
        
    """
    def __init__(self, n: int, m:int, c: List[float], 
                 a: float, b: float) -> 'Chebyshev': 
        self.n = n
        self.m = m
        self.c = c 
        self.a = a
        self.b = b
        
    @classmethod
    def from_coef(cls, coef_list: List[float], a: float,
                  b:float) -> 'Chebyshev': 
        """ Creates a chebyshev polynomial approximation from previously
            computed coefficients
            
        Args: 
            coef_list: pre-computed coefficient list
            a: start of fit interval
            b: end of fit interval 
            
        """
        n = len(coef_list)
        m = n 
        c = coef_list
        a = a 
        b = b
        
        return cls(n, m, c, a, b)
        
        
    @classmethod
    def from_fxn(cls, fxn: Callable, a: float, b: float, n: int=50) -> 'Chebyshev':
        """ Creates a chebyshev polynomial approximation of a function
            over some interval. 
            
        Chebyshev fit: Given a function fxn, lower and upper limits of the interval
        [a, b] compute and save nn coefficients of the Chebyshev approximation. 
        This is inteded to be called with moderately large n (e.g. 30 or 50), 
        the array of c's can later be trunctated at some smaller value of m
        such that c_m and subsequent elements are negligible. 
            
        Args:
            fxn: function to approximate (1 dimensional)
            aa: beginning of approximation interval
            bb: end of approximation interval 
            nn (default=50): order of approximation 
            
        """
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        
        # pre allocate function and coefficient arrays 
        coef_list = [0 for i in range(n)]
        f_vals = [0 for i in range(n)]
        
        # evaluate the function at the N points required
        for k in range(n): 
            y = np.cos(np.pi * (k + 0.5) / n)
            f_vals[k] = fxn(y * bma + bpa)
        
        # now evaluate cheby coefficients 
        fac = 2.0 / n 
        for j in range(n): 
            sum_val = 0.0
            for k in range(n): 
                sum_val += f_vals[k] * np.cos(np.pi * j * (k + 0.5) / n)
            coef_list[j] = fac * sum_val 
        
        return cls(n, n, coef_list, a, b)
       
        
    
    def setm(self, thresh: float) -> int: 
        """ Replaces m based on an error threshold
        
        Args: 
            thresh: error threshold to use for setting m
        
        Note: Modifies object in place!
        
        """
        m = self.m
        while m > 1 and np.abs(self.c[m -1]) < thresh:
            m -= 1
            
        self.m = m
              
              
    def evaluate(self, x: float, m: int=None) -> float: 
        """ Evaluate the Chebyshev at a value with some truncated number
            of coefficients
            
        Args: 
            x: point at which to evaluate chebyshev approximation
            m: truncation degree
            
        Returns: 
            float
        
        """
        if m is None:
            m = self.m
        
        # initialize values
        d = 0.0
        dd = 0.0
              
        if (x - self.a) * (x - self.b) > 0.0: 
              raise ValueError("x not in range of Cheby Approximation!")
        
        y = (2.0 * x - self.a - self.b) / (self.b - self.a)
        y2 = 2.0 * y # change of variables 
        
        # uses the Clenshaw recurrence
        for j in [i for i in range(1, self.m)][::-1]: 
            sv = d
            d = y2 * d - dd + self.c[j]
            dd = sv
        
        # last step of the process is a little different
        return y * d - dd + 0.5 * self.c[0]
              
        
    def derivative(self): 
        """ Returns a new chebyshev object approximating the derivative
            over the same interval [a,b] as the original object
            
        """
        coef_der = [0 for i in range(self.n)]
        coef_der[self.n - 1] = 0.0
        coef_der[self.n - 2] = 2 * (self.n - 1) * self.c[self.n - 1]
        
        for j in [i for i in range(1, self.n - 2)][::-1]: 
            coef_der[j - 1] = coef_der[j + 1] + 2 * j * self.c[j]
        
        # normalizing factor
        fact_norm = 2.0 / (self.b - self.a) 
        
        for i in range(self.n): 
            coef_der[i] *= fact_norm

            
        return self.from_coef(coef_der, self.a, self.b)
        
        
    def integral(self):
        """ Returns a new chebyshev object approximating the integral
            over the same time interval as the original object
            
        """
        sum_val = 0.0
        fac = 1.0
        # factor that normalizes to the interval b-a
        coef_int = [0 for i in range(self.n)]
        con = 0.25 * (self.b - self.a)
        
        for j in range(1, self.n - 1):  
            coef_int[j] = con * (self.c[j - 1] - self.c[j + 1]) / j
            sum_val += fac * coef_int[j]
            fac = -fac
        
        coef_int[self.n - 1] = con * self.c[self.n - 2] / (self.n - 1)
        sum_val += fac * coef_int[self.n - 1]
        coef_int[0] = 2.0 * sum_val
        
        return self.from_coef(coef_int, self.a, self.b)

# Test functions
def cheby_test(x): 
    """ Runge Function """
    return 1 / (1 + 25 * x**2)

def cheby_der_test(x): 
    """ Derivative of the runge function """
    return -50 * x / (1 + 25 * x**2)**2

def cheby_int_test(x): 
    """ Integral of the runge function """
    return 1 / 5 * np.arctan(5 * x)


if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    #### test out chebyshev approximation of the Runge Function ###
    q = Chebyshev.from_fxn(cheby_test, -1, 1, 200)
    xs = [i for i in np.arange(-.9, .9, 0.01)]
    ys_test = [q.evaluate(x) for x in xs]
    ys_true = [cheby_test(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys_test)
    plt.plot(xs, ys_true)
    plt.legend(["Test", "True"])
    # plot the error relative to true
    plt.figure()
    plt.plot(xs, [y_tru - y_tst for y_tru, y_tst in zip(ys_true, ys_test)])

    ### test out derivative generation from chebyshev object ###
    p = q.derivative()
    ydots_test = [p.evaluate(x) for x in xs]
    ydots_true = [cheby_der_test(x) for x in xs]

    plt.figure()
    plt.plot(xs, ydots_test)
    plt.plot(xs, ydots_true)
    plt.legend(["Test", "True"])

    # error #
    plt.figure()
    plt.plot(xs, [y_tru - y_tst for y_tru, y_tst in zip(ydots_true, 
                                                        ydots_test)])

    ### test out integral generation 
    g = q.integral()
    yints_test = [g.evaluate(x) for x in xs]
    yints_true = [cheby_int_test(x) - cheby_int_test(-1) for x in xs]
    
    plt.plot(xs, yints_test)
    plt.plot(xs, yints_true)
    plt.legend(["Test", "True"])

    # error #
    plt.plot(xs, [y_tru - y_tst for y_tru, y_tst in zip(yints_true, 
                                                        yints_test)])