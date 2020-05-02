import numpy as np
dtype = np.float32

class PA_alg():
    def __init__(self, M, K, maxP):
        self.M = M
        self.K = K
        self.maxP = maxP
        self.sigma2 = 1e-3*pow(10., -114./10.)

    def Load_data(self, H2, p_array):
        self.H2 = H2
        self.p_array = p_array
        self.N = H2.shape[0]
        self.W = np.ones((self.N), dtype = dtype)
        
    def Calculate(self):   
        return self.FP_algorithm(), self.WMMSE_algorithm(), self.Max_power(), self.Random_power()
        
    def FP_algorithm(self):
        P = np.random.rand(self.N) # maxP*np.ones((N))
        P_extend = np.hstack([P, np.zeros((self.M - self.N + 1), dtype=dtype)])
        P_matrix = np.reshape(P_extend[self.p_array], [self.N,self.K])
        g_ii = self.H2[:,0]
        for cou in range(100):    
            P_last = P
            gamma = g_ii * P_matrix[:,0] / (np.sum(self.H2[:,1:] * P_matrix[:,1:], axis=1) + self.sigma2)
            y = np.sqrt(self.W * (1.+gamma) * g_ii * P_matrix[:,0]) / (np.sum(self.H2 * P_matrix, axis=1) + self.sigma2)    
            y_j = np.tile(np.expand_dims(y, axis=1), [1,self.K])
            P = np.minimum(self.maxP, np.square(y) * self.W * (1.+gamma) * g_ii / np.sum(np.square(y_j)*self.H2, axis=1))       
            if np.linalg.norm(P_last - P) < 1e-3:
                break
            P_extend = np.hstack([P, np.zeros((self.M - self.N + 1), dtype=dtype)])
            P_matrix = np.reshape(P_extend[self.p_array], [self.N,self.K])           
        return P
    
    def WMMSE_algorithm(self):
        hkk = np.sqrt(self.H2[:,0])
        v = np.random.rand(self.N) # maxP*np.ones((N))
        V_extend = np.hstack([v, np.zeros(((self.M - self.N + 1)), dtype=dtype)])
        V = np.reshape(V_extend[self.p_array], [self.N,self.K])    
        u = hkk*v / (np.sum(self.H2*V**2, axis=1) + self.sigma2)
        w = 1. / (1. - u * hkk * v)
        C = np.sum(w)  
        W_extend = np.hstack([w, np.zeros((self.M - self.N + 1), dtype=dtype)])
        W = np.reshape(W_extend[self.p_array], [self.N,self.K])
        U_extend = np.hstack([u, np.zeros((self.M - self.N + 1), dtype=dtype)])
        U = np.reshape(U_extend[self.p_array], [self.N,self.K])
        for cou in range(100):   
            C_last = C
            v = w*u*hkk / np.sum(W*U**2*self.H2, axis=1)
            v = np.minimum(np.sqrt(self.maxP), np.maximum(1e-10*np.random.rand(self.N), v))
            V_extend = np.hstack([v, np.zeros((self.M - self.N + 1), dtype=dtype)])
            V = np.reshape(V_extend[self.p_array], [self.N,self.K])  
            u = hkk*v / (np.sum(self.H2*V**2, axis=1) + self.sigma2)
            w = 1. / (1. - u * hkk * v)
            C = np.sum(w)     
            if np.abs(C_last - C) < 1e-3:
                break
            W_extend = np.hstack([w, np.zeros((self.M - self.N + 1), dtype=dtype)])
            W = np.reshape(W_extend[self.p_array], [self.N,self.K])
            U_extend = np.hstack([u, np.zeros((self.M - self.N + 1), dtype=dtype)])
            U = np.reshape(U_extend[self.p_array], [self.N,self.K])
        P = v**2
        return P
        
    def Max_power(self):
        P = self.maxP*np.ones((self.N))         
        return P
        
    def Random_power(self):
        P = self.maxP*np.random.rand((self.N))     
        return P
