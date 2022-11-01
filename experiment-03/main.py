from cgi import test
import numpy as np
from utils.dataset import get_data,get_paras,gaussian_prop
dataA,dataB = get_data()
mu_a, sigma_a = get_paras(dataA)
mu_b, sigma_b = get_paras(dataB)
testA,testB = get_data('./dataset/Test.txt')

proA = gaussian_prop(testA,mu_a,sigma_a)
proA_ = gaussian_prop(testA,mu_b,sigma_b)
A_acc = np.array(proA > proA_,dtype=int)
print('Accuracy in A  %.2f'%np.mean(A_acc))

proB = gaussian_prop(testB,mu_b,sigma_b)
proB_ = gaussian_prop(testB,mu_a,sigma_a)
B_acc = np.array(proB > proB_,dtype=int)
print('Accuracy in B  %.2f'%np.mean(B_acc))

total_acc = np.concatenate([A_acc,B_acc])

print('Total accurary %.2f'%np.mean(total_acc))


