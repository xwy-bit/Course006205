import numpy as np

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

def get_paras(datas):
    mu = np.mean(datas,axis=0)
    sigma  = np.matmul((datas-mu).transpose(),datas-mu)/(datas.shape[0]-1)  
    return mu , sigma      
 
def gaussian_prop(datas,mu,sigma):
    d = datas.shape[1]
    coff0 = 1.0/(np.sqrt((2*np.pi)**d * np.linalg.det(sigma)))
    sigma_ = np.linalg.inv(sigma)
    coff1 = np.exp(np.sum(-0.5*np.matmul((datas-mu),sigma_) * (datas-mu),axis=1))
    return coff0 * coff1