import numpy as np
from bisect import bisect_left
import time
from math import atan2
import scipy.special
from scipy.stats import norm

import warnings
warnings.filterwarnings("error")


def angle(x):
    if np.dot(x,x) < 1e-7:
        return 0.
    else:
        angle = atan2(x[1],x[0])
        if angle >= 0:
            return angle
        else:
            return angle + 2*np.pi

def convert_polar(x):
    theta = angle(x)
    r = np.sqrt(x[0]**2 + x[1]**2)
    return r, theta

dic = ["minus","plus"]


class BrownianMotion:

    def __init__(self,r0,theta0,T,alpha):
        self.initial_r0 = r0
        self.initial_theta0 = theta0
        self.alpha = alpha
        self.T = T
        self.T_n = 0.
        self.nb_iter = 0
        self.r0, self.theta0 = r0, theta0
        self.m = int(np.ceil(np.pi/alpha))
        self.Theta = np.pi/(np.ceil(np.pi/alpha))
        self.isOnBorder = False
        self.isApprox = False

    def reset(self):
        self.r0, self.theta0 = self.initial_r0, self.initial_theta0
        self.T_n = 0.
        self.nb_iter = 0
        self.isOnBorder = False
        self.isApprox = False


    def tau_simul(self,r,pm):
        # simulate the exit time knowing the exit radius and the exit frontier + or -
        # only if alpha-=0 et alpha+=alpha
        if pm == "plus":
            gamma_k = (np.pi/self.m) - self.theta0 + 2.*np.arange(self.m)*np.pi/self.m
        if pm == "minus":
            gamma_k = self.theta0 - 2.*np.arange(self.m)*np.pi/self.m

        gamma_k_list = np.array([gamma for gamma in gamma_k if np.sin(gamma) >= 0])
        llambda_k = np.sin(gamma_k_list)/( (r-self.r0*np.cos(gamma_k_list))**2 + (self.r0**2) * np.sin(gamma_k_list)**2 )
        llambda_accumulate = np.cumsum(llambda_k)
        llambda_accumulate = llambda_accumulate/llambda_accumulate[-1]

        bool = True
        while bool:
            index = bisect_left(llambda_accumulate,np.random.uniform()) # choose which distribution in the reference mixture
            t_sim = np.random.exponential( (((r-self.r0*np.cos(gamma_k_list[index]))**2 + self.r0**2 * np.sin(gamma_k_list[index])**2)/2.)**(-1) )**(-1) # simulate tau as inverse of exponential distribution

            if np.random.uniform() < np.sum( np.sin(gamma_k) * np.exp(-((r-self.r0*np.cos(gamma_k))**2 + (self.r0**2)*np.sin(gamma_k)**2)/(2.*t_sim) )) / np.sum( np.sin(gamma_k_list) * np.exp(-((r-self.r0*np.cos(gamma_k_list))**2 + (self.r0**2) * np.sin(gamma_k_list)**2)/(2.*t_sim) ) ):
                bool = False
        return t_sim


    def non_exit_simul(self):
        # simulate final r and theta, knowing that we do not leave the wedge
        bool = True
        while bool:
            k = np.random.randint(self.m)
            y_sim = np.array([ np.random.normal(self.r0*np.cos(self.theta0-2.*k*self.Theta),np.sqrt(self.T-self.T_n)) , np.random.normal(self.r0*np.sin(self.theta0-2.*k*self.Theta),np.sqrt(self.T-self.T_n)) ])
            r,theta = convert_polar(y_sim)

            theta_k = np.array([(-1)**k * theta + k*self.Theta - (k%2==1)*self.Theta for k in range(2*self.m)])
            theta_2k = np.array(theta_k[::2])
            try :
                if np.random.uniform() < np.sum((-1)**np.arange(2*self.m) *np.exp(r*self.r0*np.cos(self.theta0-theta_k)/(self.T-self.T_n))) * ( theta <= self.Theta and theta >= 0. ) / np.sum(np.exp(r*self.r0*np.cos(self.theta0-theta_2k)/(self.T-self.T_n))):
                    bool = False
            except RuntimeWarning:
                if ( theta <= self.Theta and theta >= 0. ):
                    bool = False
        self.r0 = r
        self.theta0 = theta
        self.T_n = self.T


    def frontier_simul(self):
        return dic[np.random.binomial(1,self.theta0/self.Theta)]


    def r_simul(self,pm):
        r_prime_0 = self.r0**(self.m/2.)
        theta_prime_0 = self.theta0*(self.m/2.)
        U = np.random.uniform()

        if pm=="minus":
            r_prime = r_prime_0 * np.sqrt(np.cos(2.*theta_prime_0) -np.sin(2.*theta_prime_0)/np.tan((np.pi-2.*theta_prime_0)*(U-1.)) )

        if pm=="plus":
            r_prime = r_prime_0 * np.sqrt(-np.cos(2.*theta_prime_0) -np.sin(2.*theta_prime_0)/np.tan(2.*theta_prime_0*(U-1.)) )

        r = r_prime**(2./self.m)
        return r


    def stopped_BM_pi_m(self, computeTau=True):
        # Case alpha = pi/m ; We assume that the borders of the wedge are theta=0 and theta=pi/m
        pm = self.frontier_simul()
        if pm == "minus":
            theta = 0.
        else :
            theta = self.Theta
        r = self.r_simul(pm)
        if computeTau:
            tau = self.tau_simul(r,pm)
        else:
            tau=0.
        return r, theta, tau


    def stopped_BM(self, stopAtT = True, computeTau = True):
        # Simulate the stopped BM for a general angle alpha and give final r, theta
        # We assume that the borders of the wedge are theta=0 and theta=alpha
        while True:
            self.nb_iter += 1
            # definition of the frontiers beta_minus and beta_plus
            if self.theta0 <= self.Theta/2.:
                beta_minus = 0.
            elif self.theta0 <= self.alpha - self.Theta/2.:
                beta_minus = self.theta0 - self.Theta/2.
            else:
                beta_minus = self.alpha - self.Theta

            # change of variable (rotation) if beta_minus is not 0
            self.theta0 -= beta_minus
            r, theta, tau = self.stopped_BM_pi_m(computeTau)

            if (stopAtT) and (self.T_n + tau > self.T - 1e-6):
                self.non_exit_simul()
                self.theta0 += beta_minus
                return False

            theta += beta_minus
            self.r0, self.theta0 = r, theta
            self.T_n += tau
            if theta < 1e-6 or theta > self.alpha-(1e-6):
                self.isOnBorder = True
                return True # in this case self.T_n is tau


    def metzler_stopped(self, computeTau = False, N_approx=25):
        # simulate W_tau, tau, according to Metzler's method
        r_prime_0 = self.r0**(np.pi/(2.*self.alpha))
        theta_prime_0 = self.theta0*(np.pi/(2.*self.alpha))
        pm = dic[np.random.binomial(1,self.theta0/self.alpha)]
        U = np.random.uniform()

        if pm=="minus":
            theta_prime = 0.
            r_prime = r_prime_0 * np.sqrt(np.cos(2.*theta_prime_0) -np.sin(2.*theta_prime_0)/np.tan((np.pi-2.*theta_prime_0)*(U-1.)) )

        if pm=="plus":
            theta_prime = np.pi/2.
            r_prime = r_prime_0 * np.sqrt(-np.cos(2.*theta_prime_0) -np.sin(2.*theta_prime_0)/np.tan(2.*theta_prime_0*(U-1.)) )

        r = r_prime**(2.*self.alpha/np.pi)
        theta = theta_prime*(2.*self.alpha/np.pi) # r and theta are the simulated value

        if computeTau:
            C=15 # estimation of the majoring constant
            while True:
                tau_sim = r * self.r0 * np.random.standard_cauchy() # sample from reference distribution (Cauchy)
                cauchy_density = ( r * self.r0 * np.pi * (1.+ (tau_sim/(r*self.r0))**2) )**(-1)

                try :
                    if pm=="minus":
                        constant_normalization = (2./(np.pi*r_prime_0)) * (r_prime/r_prime_0)*np.sin(2*theta_prime_0) / ( np.sin(2.*theta_prime_0)**2 + ((r_prime/r_prime_0)**2 - np.cos(2.*theta_prime_0) )**2)

                        true_density = (np.pi*np.exp(-(r**2+self.r0**2)/(2.*tau_sim))/(constant_normalization*(self.alpha**2)*tau_sim*r)) * np.sum(np.arange(N_approx)*np.sin(np.arange(N_approx)*np.pi*self.theta0/self.alpha)*scipy.special.iv(np.arange(N_approx)*np.pi/self.alpha,r*self.r0/tau_sim))

                    if pm=="plus":
                        constant_normalization = (2./(np.pi*r_prime_0)) * (r_prime/r_prime_0)*np.sin(2*theta_prime_0) / ( np.sin(2.*theta_prime_0)**2 + ((r_prime/r_prime_0)**2 + np.cos(2.*theta_prime_0) )**2)

                        true_density = (np.pi*np.exp(-(r**2+self.r0**2)/(2.*tau_sim))/(constant_normalization*(self.alpha**2)*tau_sim*r)) * np.sum(np.arange(N_approx)*np.sin(np.arange(N_approx)*np.pi*(self.alpha-self.theta0)/self.alpha)*scipy.special.iv(np.arange(N_approx)*np.pi/self.alpha,r*self.r0/tau_sim))

                    if np.random.uniform() < true_density/(C*cauchy_density):
                        self.r0, self.theta0 = r, theta
                        self.T_n = tau_sim
                        return None

                except RuntimeWarning:
                    self.r0, self.theta0 = r, theta
                    self.T_n = tau_sim
                    return None

        else:
            self.r0, self.theta0 = r, theta
            return None


    def reflected_BM(self, approx = False, epsilon = 0., stopNbIter = False, nb_iter_MAX = 200):
        while True:
            if stopNbIter and self.nb_iter >= nb_iter_MAX:
                return None
            self.nb_iter += 1
            if approx and self.r0**2/(self.T - self.T_n) < epsilon: # approximation
                self.theta0 = self.alpha * np.random.uniform()
                self.r0 = self.r_approx_simul()
                self.isApprox = True
                return None

            beta_minus = self.theta0 - self.Theta/2

            # Change of variable (rotation)
            self.theta0 -= beta_minus
            r, theta, tau = self.stopped_BM_pi_m()

            if self.T_n + tau >= self.T - 1e-6:
                self.non_exit_simul()
                self.theta0 += beta_minus
                self.theta0 = self.reflect_theta(self.theta0)
                return None

            theta += beta_minus
            self.theta0 += beta_minus
            theta = self.reflect_theta(theta)
            self.r0, self.theta0 = r, theta
            self.T_n += tau


    def reflect_theta(self, theta):
        if theta <= self.alpha and theta >= 0:
            return theta
        elif theta < 0:
            return -theta
        else:
            return 2*self.alpha - theta


    def r_approx_simul(self):
        # simulate r according to the approximation distribution ; we simulate it as a mixture of two distributions
        t_prime = self.T - self.T_n
        proba = t_prime * np.exp(-self.r0/(2.*t_prime)) / ( t_prime * np.exp(-self.r0/(2.*t_prime)) + self.r0 * np.sqrt(2.*np.pi*t_prime) * (1.-norm.cdf(-self.r0/np.sqrt(t_prime))) )
        index = np.random.binomial(1,proba)

        if index == 1:
            r = self.r0 + np.sqrt(self.r0**2 - 2.*t_prime*np.log(1-np.random.uniform()))
            return r
        if index == 0:
            while True:
                r = np.random.normal(self.r0, np.sqrt(t_prime))
                if r >= 0:
                    return r


    def Monte_Carlo(self, F_test, method, N, *args):
        E1 = 0.
        E2 = 0.
        t1 = time.time()
        E_nb_iter = 0.
        E_T_n = 0.
        E_T_n_2 = 0.
        nb_iter_list = []
        for n in range(N):
            self.reset()
            getattr(self, method)(*args)
            result = F_test( np.array([self.r0*np.cos(self.theta0), self.r0*np.sin(self.theta0)]) )
            E1 += (1./N) * result
            E2 += (1./N) * result**2
            E_nb_iter += (1./N) * self.nb_iter
            E_T_n += (1./N) * self.T_n
            E_T_n_2 += (1./N) * self.T_n**2
            nb_iter_list.append(self.nb_iter)
            print('Progress: {:8.2f}%'.format(100.*n/N), end='\r')
        t2 = time.time()

        Var_sim = E2 - E1**2
        interval = 1.96*np.sqrt(Var_sim/N)
        Var_T_n = np.abs(E_T_n_2 - E_T_n**2)
        interval_T_n = 1.96*np.sqrt(Var_T_n/N)
        t_delta = t2-t1
        print('Expectation is {:8.4f} +- {:8.4f} with simulation time {:8.4f} and average number of iterations {:8.4f}'.format(E1, interval, t_delta, E_nb_iter))
        print('Simulated variance is {:8.4f}'.format(Var_sim))
        print('Simulated average min(tau,T) is {:8.4f} +- {:8.4f}\n'.format(E_T_n, interval_T_n))
        return {'E1':E1, 'interval':interval, 'E_nb_iter':E_nb_iter, 'nb_iter_list':nb_iter_list, 'E_T_n':E_T_n, 'interval_T_n':interval_T_n}





class Process: # abstract class

    def euler_scheme(self):
        raise NotImplementedError('subclasses must override euler_scheme !')

    def reset(self):
        raise NotImplementedError('subclasses must override reset !')

    def Monte_Carlo(self, F_test, N):
        E1 = 0.
        E2 = 0.
        t1 = time.time()
        for n in range(N):
            self.reset()
            self.euler_scheme()
            result = F_test(self.x0)
            E1 += (1./N) * result
            E2 += (1./N) * result**2
            print('Progress: {:8.2f}%'.format(100.*n/N), end='\r')
        t2 = time.time()

        Var_sim = E2 - E1**2
        interval = 1.96*np.sqrt(Var_sim/N)
        t_delta = t2-t1
        print('Expectation is {:8.4f} +- {:8.4f} with simulation time {:8.4f}'.format(E1, interval, t_delta))
        print('Simulated variance is {:8.4f}\n'.format(Var_sim))
        return {'E1':E1, 'interval':interval}


# reflected process with sigma = I_2
class ReflectedProcess(Process):

    def __init__(self, x0, T, alpha, b, nb_iter_euler, epsilon):
        self.initial_x0 = x0 # two-dimensional
        self.x0 = x0
        self.alpha = alpha
        self.T = T
        self.b = b
        self.h = T/nb_iter_euler
        self.BM = BrownianMotion(0., 0., self.h, alpha)
        self.nb_iter_euler = nb_iter_euler
        self.epsilon = epsilon

    def reset(self):
        self.x0 = self.initial_x0
    
    def set_BM(self):
        self.BM.r0, self.BM.theta0 = convert_polar(self.x0)
        self.BM.T_n = 0.

    def drift_step(self):
        drift = self.b(self.x0)
        x0_new = self.x0 + self.h * drift

        if drift[1] != 0.:
            h_1 = -self.x0[1]/drift[1]
        else:
            h_1 = -1.
        if np.dot(drift, np.array([np.sin(self.alpha), -np.cos(self.alpha)])) != 0:
            h_2 = - np.dot(self.x0, np.array([np.sin(self.alpha), -np.cos(self.alpha)])) / np.dot(drift, np.array([np.sin(self.alpha), -np.cos(self.alpha)]))
        else:
            h_2 = -1.
        # h_i is the time for the linear process to reach the ith barrier

        if (h_1 >= 0.) and (h_1 <= self.h) and ((self.x0+h_1*drift)[0]>0):
            self.x0 = np.array([x0_new[0],0.])
        elif (h_2 >= 0.) and (h_2 <= self.h) and ((self.x0+h_2*drift)[0]>0):
            self.x0 = np.dot(x0_new, np.array([np.cos(self.alpha), np.sin(self.alpha)])) * np.array([np.cos(self.alpha), np.sin(self.alpha)])
        else:
            self.x0 = x0_new

        if self.x0[0] < 0.:
            self.x0 = np.zeros(2)

    def brownian_step(self):
        self.set_BM()
        self.BM.reflected_BM(approx=True, epsilon=self.epsilon)
        self.x0 = np.array([self.BM.r0 * np.cos(self.BM.theta0), self.BM.r0 * np.sin(self.BM.theta0)])
    
    def euler_scheme(self):
        for k in range(self.nb_iter_euler):
            self.drift_step()
            self.brownian_step()



# stopped process ; we only do sigma = I_2 for now
class StoppedProcess(Process):

    def __init__(self, x0, T, alpha, b, nb_iter_euler):
        self.initial_x0 = x0 # two-dimensional
        self.x0 = x0
        self.alpha = alpha
        self.T = T
        self.b = b
        self.h = T/nb_iter_euler
        self.nb_iter_euler = nb_iter_euler
        self.BM = BrownianMotion(0., 0., self.h, alpha)
        self.isOnBorder = False

    def reset(self):
        self.x0 = self.initial_x0
        self.isOnBorder = False
    
    def set_BM(self):
        self.BM.r0, self.BM.theta0 = convert_polar(self.x0)
        self.BM.T_n = 0.

    def drift_step(self):
        drift = self.b(self.x0)
        x0_new = self.x0 + self.h * drift

        if drift[1] != 0.:
            h_1 = -self.x0[1]/drift[1]
        else:
            h_1 = -1.
        if np.dot(drift, np.array([np.sin(self.alpha), -np.cos(self.alpha)])) != 0:
            h_2 = - np.dot(self.x0, np.array([np.sin(self.alpha), -np.cos(self.alpha)])) / np.dot(drift, np.array([np.sin(self.alpha), -np.cos(self.alpha)]))
        else:
            h_2 = -1.
        # h_i is the time for the linear process to reach the ith barrier

        if (h_1 >= 0.) and (h_1 <= self.h) and ((self.x0+h_1*drift)[0]>0):
            self.x0 = self.x0 + h_1*drift
            self.isOnBorder = True
        elif (h_2 >= 0.) and (h_2 <= self.h) and ((self.x0+h_2*drift)[0]>0):
            self.x0 = self.x0 + h_2*drift
            self.isOnBorder = True
        else:
            self.x0 = x0_new

    def brownian_step(self):
        self.set_BM()
        self.isOnBorder = self.BM.stopped_BM()
        self.x0 = np.array([self.BM.r0 * np.cos(self.BM.theta0), self.BM.r0 * np.sin(self.BM.theta0)])
    
    def euler_scheme(self):
        for k in range(self.nb_iter_euler):
            self.drift_step()
            if self.isOnBorder:
                return None
            self.brownian_step()
            if self.isOnBorder:
                return None
        return None


