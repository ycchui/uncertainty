import matplotlib.pyplot as plt
import numpy as np
import os

#directory of model suite
store_path = "2nd_gen_Nsingle_type3"

#load path
repo_path  =  os.path.realpath("")
#os.chdir(os.path.join(repo_path,store_path))

#data to read (deepest, shallowest, volume, neighbour)
type = ['volume','deepest','shallowest','neighbour']


#read data
def read_data(name, shape):
    return np.loadtxt(name+'.txt',).reshape(100,shape)

#print histogram
def print_hist(data, title):
    hist, bins = np.histogram(data, bins=10)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    ax.set_title(title)
    ax.set_ylabel('frequency')
    fig.savefig(title)
    plt.close()

for metric in type:
    #reload path
    os.chdir(os.path.join(repo_path,store_path))
    #which file to read
    name = 'unit_'+metric
    save_path = metric

    data = read_data(name,7)
    #print data
    if not os.path.exists(os.path.join(os.getcwd(), save_path)):
        os.makedirs(os.path.join(os.getcwd(),save_path))
    os.chdir(os.path.join(os.getcwd(),save_path))

    for unit in range(0,7,1):
        print_hist(data[:,unit],'unit'+str(unit+1)+'_'+metric)
