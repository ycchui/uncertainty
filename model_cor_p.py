import matplotlib.pyplot as plt
import numpy as np
import os
import math
from pandas.tools.plotting import scatter_matrix
import pandas as pd

#directory of model suite
store_path = "2nd_gen_Nsingle_type3"

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))
'''metrics = ['unit_volume','unit_deepest','unit_shallowest','unit_neighbour','contact_surface_area','contact_curvature_avg_gauss','contact_curvature_avg_mean',\
'contact_curvature_avg_std_gauss','contact_curvature_avg_std_mean','contact_curvature_sum_gauss','contact_curvature_sum_mean',\
'contact_curvature_sum_std_gauss','contact_curvature_sum_std_mean']
metrics_no = [7,7,7,7,6,6,6,6,6,6,6,6,6]'''
#excluded deepest and shallowest
metrics = ['unit_volume','unit_neighbour','contact_surface_area','contact_curvature_avg_gauss','contact_curvature_avg_mean',\
'contact_curvature_avg_std_gauss','contact_curvature_avg_std_mean','contact_curvature_sum_gauss','contact_curvature_sum_mean',\
'contact_curvature_sum_std_gauss','contact_curvature_sum_std_mean']
metrics_no = [7,7,6,6,6,6,6,6,6,6,6]
#read data
def read_data(name,shape):
    return np.loadtxt(name+'.txt',).reshape(100,shape)
    
def save_result(name, result):
    with file(name+".txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(result.shape))
        outfile.write('# File format: unit1, unit2, unit3, ...\n')
        np.savetxt(outfile, result)

def wrap(txt, width=8):
    '''helper function to wrap text for long labels'''
    import textwrap
    return '\n'.join(textwrap.wrap(txt, width))


t = [['u1_volume','u2_volume','u3_volume','u4_volume','u5_volume','u6_volume','u7_volume'],
#['u1_deepest','u2_deepest','u3_deepest','u4_deepest','u5_deepest','u6_deepest','u7_deepest'],
#['u1_shallowest','u2_shallowest','u3_shallowest','u4_shallowest','u5_shallowest','u6_shallowest','u7_shallowest'],
['u1_neighbour','u2_neighbour','u3_neighbour','u4_neighbour','u5_neighbour','u6_neighbour','u7_neighbour'],
['u1/2_surface','u2/3_surface','u3/4_surface','u4/5_surface','u5/6_surface','u6/7_surface'],
['u1/2_surface_cur_gauss_avg','u2/3_surface_cur_gauss_avg','u3/4_surface_cur_gauss_avg',\
'u4/5_surface_cur_gauss_avg','u5/6_surface_cur_gauss_avg','u6/7_surface_cur_gauss_avg'],
['u1/2_surface_cur_mean_avg','u2/3_surface_cur_mean_avg','u3/4_surface_cur_mean_avg',\
'u4/5_surface_cur_mean_avg','u5/6_surface_cur_mean_avg','u6/7_surface_cur_mean_avg'],
['u1/2_surface_cur_gauss_avg_std','u2/3_surface_cur_gauss_avg_std','u3/4_surface_cur_gauss_avg_std',\
'u4/5_surface_cur_gauss_avg_std','u5/6_surface_cur_gauss_avg_std','u6/7_surface_cur_gauss_avg_std'],
['u1/2_surface_cur_mean_avg_std','u2/3_surface_cur_mean_avg_std','u3/4_surface_cur_mean_avg_std',\
'u4/5_surface_cur_mean_avg_std','u5/6_surface_cur_mean_avg_std','u6/7_surface_cur_mean_avg_std'],
['u1/2_surface_cur_gauss_sum','u2/3_surface_cur_gauss_sum','u3/4_surface_cur_gauss_sum',\
'u4/5_surface_cur_gauss_sum','u5/6_surface_cur_gauss_sum','u6/7_surface_cur_gauss_sum'],
['u1/2_surface_cur_mean_sum','u2/3_surface_cur_mean_sum','u3/4_surface_cur_mean_sum',\
'u4/5_surface_cur_mean_sum','u5/6_surface_cur_mean_sum','u6/7_surface_cur_mean_sum'],
['u1/2_surface_cur_gauss_sum_std','u2/3_surface_cur_gauss_sum_std','u3/4_surface_cur_gauss_sum_std',\
'u4/5_surface_cur_gauss_sum_std','u5/6_surface_cur_gauss_sum_std','u6/7_surface_cur_gauss_sum_std'],
['u1/2_surface_cur_mean_sum_std','u2/3_surface_cur_mean_sum_std','u3/4_surface_cur_mean_sum_std',\
'u4/5_surface_cur_mean_sum_std','u5/6_surface_cur_mean_sum_std','u6/7_surface_cur_mean_sum_std']]
for metric in range(0,11,1):
    
    #data = np.hstack((data, read_data('unit_deepest')))
    for metric2 in range(metric+1,11,1):
        data = read_data(metrics[metric],metrics_no[metric])
        data = np.hstack((data, read_data(metrics[metric2],metrics_no[metric2])))
        #data = data.transpose()

        title = t[metric]+t[metric2]
        #print data
        print data.shape
        print title

        df = pd.DataFrame(data, columns = title)
        axs = scatter_matrix(df, alpha=1, figsize=(60,60),marker='o', diagonal='hist')
        '''for ax in axs:
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)'''
        for ax in axs[:,0]:
            ax.set_ylabel(wrap(ax.get_ylabel()),rotation=0,va='center',fontsize=40,labelpad=110)
        for ax in axs[-1,:]:
            ax.set_xlabel(ax.get_xlabel(),rotation=90,fontsize=40)
        plt.tight_layout()
        #plt.suptitle('Correlation Matrix of '+metrics[metric]+' and '+metrics[metric2])
        plt.savefig('Correlation Matrix of '+metrics[metric]+' and '+metrics[metric2])
        plt.close()