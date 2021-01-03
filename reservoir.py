import numpy as np
import math
import logging

#logging.basicConfig(level=logging.DEBUG)
class Readout():
    def __init__(self, mode, n_in, n_out, dt):
        self.mode = mode
        self.n_in = n_in
        self.n_out = n_out
        self.dt = dt

        mu = 0.0
        sig_out = 1.0/math.sqrt(self.n_in)
        self.W = np.random.normal(loc=mu,scale=sig_out,size=(self.n_out,self.n_in))
        self.W_fb = np.random.uniform(low=-1,high=1,size=(self.n_in,self.n_out))
        self.mask_fb = 1.0

        if mode == "FORCE":
            self.alpha = 1.0 if self.n_in >= 10 else self.n_in*0.1
            self.inv_Corr = np.identity(self.n_in)/self.alpha
        elif mode == "Q":
            self.explo_noise = np.zeros((self.n_out,1))
            self.act_f = np.tanh
            self.lr = 0.001
            self.gamma = 0.99
            self.tau = 20 #20 ms
            self.sig = 0.1 #exploration noise
            self.rew = 0.0
            self.cum_rew = 0.0

        self.delta_t = 5.0 #5ms period for FORCE learning
        self.counter = self.delta_t

        self.output = np.zeros((self.n_out,1))
        self.fb = np.zeros((self.n_in,1))
    
    def learning(self, state, lr_signal):
        self.counter += self.dt
        if self.mode == "FORCE":
            if self.counter >= self.delta_t:
                err = np.squeeze(self.output - lr_signal)[()]
                tmp = np.dot(self.inv_Corr,state)
                self.inv_Corr = self.inv_Corr - np.dot(tmp,np.dot(state.T,self.inv_Corr))/(1 + np.dot(state.T,tmp))
                self.W = self.W - err * np.dot(self.inv_Corr,state).T
                self.counter = 0.0
        elif self.mode == "Q":
            self.rew += lr_signal
            self.cum_rew += lr_signal
            if self.counter >= self.delta_t:
                self.explo_noise = (1 - self.delta_t/self.tau) * self.explo_noise + self.sig * np.random.normal(loc=0.0,scale=1.0,size=(self.n_out,1))
                td = self.rew + self.gamma * (self.output + self.explo_noise) - self.output
                for ind_a in range(self.n_out):
                    for ind_j in range(self.n_in):
                        self.W[ind_a][ind_j] = self.W[ind_a][ind_j] - self.lr * self.act_f(td[ind_a]) * state[ind_j]
                self.rew = 0.0
                self.counter = 0.0

    def update(self, state):
        self.output = np.dot(self.W,state)
        self.fb = self.mask_fb * np.dot(self.W_fb,self.output)

    def get_fb(self):
        return self.fb

    def get_output(self):
        return self.output

    def reset(self):
        self.rew = 0.0
        self.cum_rew = 0.0
        self.counter = self.delta_t
        self.output = np.zeros(shape=self.output.shape)
        self.fb = np.zeros(shape=self.fb.shape)

class ESN():
    """
    docstring
    """
    def __init__(self,N,N_in):
        self.N = N
        self.N_in = N_in
        self.activation_F = np.tanh

        self.X = np.zeros((self.N,1)) #internal neuron states
        self.R = self.activation_F(self.X)
        self.tau = 10.0 #10ms
        self.dt = 1.0 #1ms

        self.I_0 = 0.001

        g_str = 1.0
        p_c = 0.1

        mu = 0.0
        sig_in = 1.0
        sig_rc = g_str/math.sqrt(p_c*self.N)

        self.W_in = np.random.normal(loc=mu,scale=sig_in,size=(self.N,self.N_in))
        self.W_rc = np.random.normal(loc=mu,scale=sig_rc,size=(self.N,self.N))

        self.readouts = []
        self.modes = ["FORCE","Q"]

    def noise(self):
        return np.random.normal(loc=0.0,scale=self.I_0,size=(self.N,1))

    def add_readout(self, mode, n_out): #initialize readout and corresponding feedback weights
        self.readouts.append(Readout(mode,self.N,n_out,self.dt))
        self.update_routs()

    def get_fb(self): #calculate the total weighted feedback current coming from all the readouts
        fb = np.zeros(shape=self.X.shape)
        for rout in self.readouts:
            fb += rout.get_fb()
        return fb

    def input(self,inp):
        if hasattr(inp,"shape"):
            assert inp.shape == (self.W_in.shape[1],1)
        self.inp = inp

    def update_routs(self):    
        for rout in self.readouts:
            rout.update(self.R)

    def step(self):
        logging.info(self.inp)
        dX = (-self.X + np.dot(self.W_rc,self.R) + np.dot(self.W_in,self.inp) + self.get_fb() + self.noise()) * self.dt / self.tau
        logging.info(dX)
        self.X += dX
        self.R = self.activation_F(self.X)
        self.update_routs()
        # self.learning()

    def learning(self):
        for rout in self.readouts:
            lr_signal = 0.0
            if rout.mode == "FORCE":
                lr_signal = self.inp
            elif rout.mode == "Q":
                lr_signal = self.rew
            rout.learning(self.R, lr_signal)

    def set_rew(self, rew):
        self.rew = rew

    def reset(self):
        self.X = np.zeros(shape=self.X.shape)
        self.R = self.activation_F(self.X)
        for rout in self.readouts:
            rout.reset()

    pass