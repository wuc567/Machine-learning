import scipy.io as sio
import numpy as np
import os
dict_data  = sio.loadmat("D:\\pythonsitem\\gcForestCS\\dataset\\sEMG\\Database 1\\female_1")
def read_mat(dict_data):
    j = 0
    data = []
    label = []
    for (i,da) in dict_data.items():
        if j>2:
            if len(data) == 0:
                data = da
                label = np.full((da.shape[0],1),int((j-3)//2))
            else:
                data = np.vstack((data,da))
                label = np.vstack((label,np.full((da.shape[0],1), int((j - 3) // 2))))
        j+=1

    return data,label


def process():
    x = []
    y = []
    for name in os.listdir("D:\\pythonsitem\\gcForestCS\\dataset\\sEMG\\Database 1"):
        dict_data = sio.loadmat("D:\\pythonsitem\\gcForestCS\\dataset\\sEMG\\Database 1\\{}".format(name))
        X,Y = read_mat(dict_data)
        if len(x) ==0:
            x = X
            y = Y
        else:

            x = np.vstack((x,X))
            y = np.vstack((y,Y))
    return x,y

print(process())
