import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import numpy as np
import os

#directory of model suite
store_path = "2nd_gen_Nsingle_type0"

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))

metrics = ['unit_volume','unit_neighbour','contact_surface_area','contact_curvature_avg_gauss','contact_curvature_avg_mean',\
'contact_curvature_avg_std_gauss','contact_curvature_avg_std_mean','contact_curvature_sum_gauss','contact_curvature_sum_mean',\
'contact_curvature_sum_std_gauss','contact_curvature_sum_std_mean']
metrics_no = [7,7,6,6,6,6,6,6,6,6,6]
t = ['u1_volume','u2_volume','u3_volume','u4_volume','u5_volume','u6_volume','u7_volume',
#'u1_deepest','u2_deepest','u3_deepest','u4_deepest','u5_deepest','u6_deepest','u7_deepest',
#'u1_shallowest','u2_shallowest','u3_shallowest','u4_shallowest','u5_shallowest','u6_shallowest','u7_shallowest',
'u1_neighbour','u2_neighbour','u3_neighbour','u4_neighbour','u5_neighbour','u6_neighbour','u7_neighbour',
'u1/2_surface','u2/3_surface','u3/4_surface','u4/5_surface','u5/6_surface','u6/7_surface',
'u1/2_surface_cur_gauss_avg','u2/3_surface_cur_gauss_avg','u3/4_surface_cur_gauss_avg',\
'u4/5_surface_cur_gauss_avg','u5/6_surface_cur_gauss_avg','u6/7_surface_cur_gauss_avg',
'u1/2_surface_cur_mean_avg','u2/3_surface_cur_mean_avg','u3/4_surface_cur_mean_avg',\
'u4/5_surface_cur_mean_avg','u5/6_surface_cur_mean_avg','u6/7_surface_cur_mean_avg',
'u1/2_surface_cur_gauss_avg_std','u2/3_surface_cur_gauss_avg_std','u3/4_surface_cur_gauss_avg_std',\
'u4/5_surface_cur_gauss_avg_std','u5/6_surface_cur_gauss_avg_std','u6/7_surface_cur_gauss_avg_std',
'u1/2_surface_cur_mean_avg_std','u2/3_surface_cur_mean_avg_std','u3/4_surface_cur_mean_avg_std',\
'u4/5_surface_cur_mean_avg_std','u5/6_surface_cur_mean_avg_std','u6/7_surface_cur_mean_avg_std',
'u1/2_surface_cur_gauss_sum','u2/3_surface_cur_gauss_sum','u3/4_surface_cur_gauss_sum',\
'u4/5_surface_cur_gauss_sum','u5/6_surface_cur_gauss_sum','u6/7_surface_cur_gauss_sum',
'u1/2_surface_cur_mean_sum','u2/3_surface_cur_mean_sum','u3/4_surface_cur_mean_sum',\
'u4/5_surface_cur_mean_sum','u5/6_surface_cur_mean_sum','u6/7_surface_cur_mean_sum',
'u1/2_surface_cur_gauss_sum_std','u2/3_surface_cur_gauss_sum_std','u3/4_surface_cur_gauss_sum_std',\
'u4/5_surface_cur_gauss_sum_std','u5/6_surface_cur_gauss_sum_std','u6/7_surface_cur_gauss_sum_std',
'u1/2_surface_cur_mean_sum_std','u2/3_surface_cur_mean_sum_std','u3/4_surface_cur_mean_sum_std',\
'u4/5_surface_cur_mean_sum_std','u5/6_surface_cur_mean_sum_std','u6/7_surface_cur_mean_sum_std']

#read data
def read_data(name,shape):
    return np.loadtxt(name+'.txt',).reshape(100,shape)

#centre spines
def centre_spines(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

#title = t1+t2
    
data = read_data(metrics[0],metrics_no[0])
#data = np.hstack((data, read_data('unit_shallowest')))
for metric in range(1,11,1):
    data = np.hstack((data, read_data(metrics[metric],metrics_no[metric])))
para = np.loadtxt('fold_parameter_change.txt',).reshape(100,6)
#print para.shape[-1]
combine = np.hstack((data[:,2],data[:,6],data[:,7],data[:,13])).reshape(100,4)

#print combine

results = PCA(data)
#print results.numcols
row_std = np.std(data, axis=0)
print results.Wt.shape
#print results.sigma
#print row_std
#print results.Wt[0]
#print results.Wt[1]
#print results.fracs #contribution of each axes
loading1 = results.Wt[0]/row_std
loading2 = results.Wt[1]/row_std
if not os.path.exists(os.path.join(os.getcwd(), 'pca')):
    os.makedirs(os.path.join(os.getcwd(),'pca'))
os.chdir(os.path.join(os.getcwd(),'pca'))
para_project = results.project(para)
print para_project.shape
fig = plt.figure()
fig.set_size_inches(10,10)
ax1 = plt.subplot(121, aspect='equal')
plt.plot(results.Y[:,0],results.Y[:,1],'o', color='blue', label = 'models')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.legend()
centre_spines(ax1)
ax2 = plt.subplot(122,aspect='equal')
plt.plot(loading1,loading2,'^', color='red',label = 'metrics')
for label, x,y in zip(t, loading1,loading2):
    if(x == np.max(loading1) or x == np.min(loading1) or y == np.max(loading2) or y == np.min(loading2)):
        plt.annotate(label, xy=(x,y))
    #plt.annotate(label, xy=(x,y))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
centre_spines(ax2)
plt.tight_layout(pad=5.0)
#plt.title('PCA of neighbourhood of lithological units')
fig.suptitle('PCA of Geodiversity Metrics')
plt.savefig('PCA of Geodiversity Metrics')
plt.show()
plt.close()

#Significance Bar Chart
ind = np.arange(0,100,10)
pc_name = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
fig, ax = plt.subplots()
plt.bar(ind,results.fracs[:10],width=10)
plt.xlabel('Principal Components')
plt.ylabel('Significance')
plt.title('Significance of principal Components')
plt.xticks(ind+5,pc_name)
plt.tight_layout()
plt.savefig('Significance of principal Components')

