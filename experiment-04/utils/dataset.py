import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

def get_data(data_path = './dataset/Train.txt'):
    file = open(data_path,'r')
    Adatas = list()
    Bdatas = list()
    lines = file.readlines()
    for index,line in enumerate(lines):
        try:
            line_number = line[5:-2]
            line_array = [float(num) for num in line_number.split()]
            if line[0] == 'A':
                Adatas.append(line_array)
            elif line[0] == 'B':
                Bdatas.append(line_array)
            else:
                raise ValueError(f'unexpected type:{line[0]}')
        except:
            print(f'[WARNING]:read line:{index} ERROR,please check manually!')
    return np.array(Adatas),np.array(Bdatas)

def get_paras(datas,K = 2,steps = 100):
    # E-STEP
    # dimensionality of data
    d = 3

    # initialize parameters
    a = np.random.randn(K)
    mu = np.random.randn(K,d)
    sigma =  np.random.randn(K,d,d)
    N = datas.shape[0]
    coef = np.zeros([K,N])
    omega = np.zeros([K,N])

    for step in range(steps):
        coef0 = a / np.sqrt(np.abs(((2*np.pi)**d) * np.linalg.det(sigma))) # SHAPE [K]
        sigma_ = np.linalg.inv(sigma)
        for kk in range(K):
            for nn in range(N):
                coef[kk,nn] = coef0[kk] * np.exp(-0.5 * (datas[nn] - mu[kk]).reshape([1,d]) @ sigma_[kk] @ (datas[nn] - mu[kk]).reshape([d,1]))
        for kk in range(K):
            for nn in range(N):
                omega[kk,nn] = a[kk] * coef[kk,nn] / np.sum(a * coef[:,nn])
        
        a_update = np.sum(omega,axis=1) / N
        mu_update = np.sum(omega.reshape([K,N,1])*datas.reshape([1,N,d]),axis=1) / np.sum(omega,axis=1).reshape([K,1])
        datas_zerobias = datas.reshape([1,N,d,1]) - mu_update.reshape([K,1,d,1])
        sigma_update = np.sum(omega.reshape([K,N,1,1]) * ( datas_zerobias @ datas_zerobias.reshape([K,N,1,d])),axis=1) * N / a_update.reshape([K,1,1])

        a = a_update
        mu = mu_update
        sigma = sigma_update


def gaussian(X, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(X)
def gmm_em(Y, K, iters):
    kmeans = KMeans(
        n_clusters=K,
        init='random',
        n_init=1,
        max_iter=1)
    kmeans.fit(Y)

    X = Y
    N, D = X.shape
    
    #Init
    alpha = np.ones((K,1)) / K         
    mu = kmeans.cluster_centers_
    cov = np.array([np.eye(D)] * K)    

    omega = np.zeros((N, K))

    for i in range(iters):

        #E-Step
        p = np.zeros((N, K))
        for k in range(K):
            p[:, k] = alpha[k] * gaussian(X, mu[k], cov[k])
        sumP = np.sum(p, axis=1)
        omega = p / sumP[:, None]

        #M-Step
        omega_sum = np.sum(omega, axis=0) 
        alpha = omega_sum / N          
        for k in range(K):
            omegaX = X * omega[:, [k]]  
            mu[k] = np.sum(omegaX, axis=0) / omega_sum[k]  

            X_mu_k = X- mu[k]                                 
            omega_X_mu_k =omega[:, [k]] * X_mu_k                    
            cov[k] = np.dot(np.transpose(omega_X_mu_k), X_mu_k) / omega_sum[k]     
    return omega, alpha, mu, cov



if __name__ == '__main__':
    
    dataA,_ = get_data('../dataset/Train.txt')
    _,alpha,mu,cov =  gmm_em(dataA,4,10)
    print(f'mixing coeficient:\n{alpha}')
    print(f'mu :\n{mu}')
    print(f'Covariance matirx:\n{cov}')