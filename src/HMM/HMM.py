import numpy as np

class HMM:
    """
    Attributes:
    A: 状态转移概率矩阵 (#types,#types)
    B: 观测概率矩阵 给定状态,产生该观测的概率 (#types,#observations)
    Pi: 初始状态分布向量
    """
    def __init__(self,A,B,Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def _forward(self, observations):
        """
        前向算法:
        计算给定HMM模型参数和状态序列的情况下
        预测该观察序列产生的概率
        """
        n_states = self.A.shape[0]
        n_samples = len(observations)
        F = np.zeros(n_states,n_samples)
        F[:,0] = self.Pi*self.B[:,observations[0]]
        
        for t in range(1,n_states):
            for n in range(n_states):
                F[n,t] = np.dot(F[:,t-1],(self.A[:,n])) * self.B[n,observations[t]]
        return F

    def _backward(self, observations):
        """
        后向算法:
        计算给定HMM模型参数和状态序列的情况下
        预测该观察序列产生的概率
        """
        n_states = self.A.shape[0]
        n_samples = len(observations)
        X = np.zeros(n_states,n_samples)
        X[:,-1:] = 1

        for t in reversed(range(n_samples-1)):
            for n in range(n_states):
                X[n,t] = np.sum(X[:,t+1]*self.A[n,:]*self.B[:,observations[t+1]])
        return X

    def baum_welch_train(self, observations, criterion=0.05):
        """
        Baum-Weich算法
        给定观测序列,学习HMM模型参数参数
        """
        n_states = self.A.shape[0]
        n_samples = len(observations)
        done = False

        while not done:
            alpha = self._forward(observations)
            beta = self._backward(observations)
            xi = np.zeros((n_states,n_states,n_samples-1)) 
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T,self.A)*self.B[:,observations[t+1]].T,beta[:,t+1])
                for i in range(n_states):
                    number = alpha[i,t]*self.A[i,:]*self.B[:,observations[t+1]].T*beta[:,t+1].T
            gamma = np.sum(xi,axis=1)
            prod = (alpha[:,n_samples-1]*beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,prod/np.sum(prod)))

            newPi = gamma[:,0]
            newA = np.sum(xi,2)/np.sum(gamma[:,:,-1],axis=1).reshape((-1,1))
            newB = np.copy(self.B)

            num_levels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1)/sumgamma
            if np.max(abs(self.Pi-newPi))<criterion and np.max(abs(self.A-newA))<criterion and np.max(abs(self.B-newB))<criterion:
                done = 1
            self.A[:],self.B[:],self.Pi[:] = newA,newB,newPi

    def observation_prob(self,observations):
        """
        整个观测序列的产生概率
        """
        return np.sum(self._forward(observations)[:,-1])

    def viterbi(self,observations):
        """
        维特比译码算法
        返回
        矩阵V V[s][t]表示max P(o1,..,ot,q1,q2,...,qt-1,qt=s|Pi,A,B)
        矩阵preV: 指向t-1时刻的状态使得V[state][t]最大
        """
        n_states = self.A.shape[0]
        n_samples = len(observations)
        preV = np.zeros((n_states-1,n_samples),dtype=int)
        V = np.zeros((n_states,n_samples))
        V[:,0] = self.Pi*self.B[:,observations[0]]
        for t in range(1,n_samples):
            for n in range(n_states):
                seq_probs = V[:,t-1]*self.A[:,n]*self.B[n,observations[t]]
                preV[t-1,n] = np.argmax(seq_probs)
                V[n,t] = np.max(seq_probs)
        return V,preV

    def build_viterbi_path(self,preV,last_state):
        """
        找到最优路径
        """
        T = len(preV)
        yield(last_state)
        for i in range(T-1,-1,-1):
            yield(preV[i,last_state])
            last_state = preV[i,last_state]

    def state_path(self,observations):
        """
        返回
        V[last_state,-1]: float 最优状态路径的概率
        path: list(int) 给定观测序列时的最优状态序列
        用于给定模型参数、观测序列时,得到其状态序列
        """
        V, preV = self.viterbi(observations)
        last_state = np.argmax(V[:,-1])
        path = list(self.build_viterbi_path(preV,last_state))
        return V[last_state,-1], reversed(path)