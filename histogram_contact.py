import matplotlib.pyplot as plt
import numpy as np
import os

#directory of model suite
store_path = "2nd_gen_Nsingle_type0"

#load path
repo_path  =  os.path.realpath("")
#os.chdir(os.path.join(repo_path,store_path))

#data to read (deepest, shallowest, volume, neighbour)
type = ['surface_area','curvature_avg','curvature_avg_std','curvature_sum','curvature_sum_std']


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
    name = 'contact_'+metric
    save_path = metric
    if metric == 'surface_area':
        data = read_data(name,6)
    else:
        data = read_data(name,12)
    #print data
    if not os.path.exists(os.path.join(os.getcwd(), save_path)):
        os.makedirs(os.path.join(os.getcwd(),save_path))
    os.chdir(os.path.join(os.getcwd(),save_path))

    if metric == 'surface_area':
        for unit in range(1,7,1):
            print_hist(data[:,unit-1],'contact_'+str(unit)+'&'+str(unit+1)+'_'+metric)
    else:
        for unit in range(1,7,1):
            print_hist(data[:,unit*2-2],'contact_'+str(unit)+'&'+str(unit+1)+'_'+metric+'_Gaussian')
            print_hist(data[:,unit*2-1],'contact_'+str(unit)+'&'+str(unit+1)+'_'+metric+'_Mean')
