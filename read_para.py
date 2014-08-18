import matplotlib.pyplot as plt
import numpy as np
import os

#directory of model suite
store_folder = ["2nd_gen_Nsingle_type0","2nd_gen_Nsingle_type1","2nd_gen_Nsingle_type2","2nd_gen_Nsingle_type3"]



no_fold = 2
no_para = 3

#read data
def read_data(name, y,z):
    return np.loadtxt(name+'.txt',).reshape(100,y,z)

#print para
def print_para(para, model):
    print 'fold 1: dip     dip dir     pitch'
    print para[model-1,0,:]
    print 'fold 2: dip     dip dir     pitch'
    print para[model-1,1,:]
    
type = raw_input("Enter Interference Type: ")
store_path = store_folder[int(float(type))]
#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))
para = read_data('fold_parameter_change',no_fold,no_para)
#para2 = np.loadtxt('fold_parameter_change.txt',).reshape(100,6)

while True:
    var = raw_input("Enter model number: ")
    if var == "quit":
        break
    else:
        print_para(para,int(float(var)))
        #print para2[int(float(var)),:]
        #print para[int(float(var)),:]

