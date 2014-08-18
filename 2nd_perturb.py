
# coding: utf-8

#This script modifies history file based on events.
#Each event can contain more than one fold.
#User need to identify number of folds in each event to let the script works correctly.

# Initializing variable
# 

# In[1]:

import pynoddy
import os
import random
import  matplotlib.pyplot  as  plt
import numpy as np

#rnd_seed = 11
#rnd_seed2 = 97

repo_path  =  os.path.realpath("")
#rnd = random
#rnd2 = random
#rnd.seed(rnd_seed)
#rnd2.seed(rnd_seed2)

rnd_list = []
rnd_seed_list = [11,97]

#perturb variable, start and end for the range [start,end), diff for increment
'''not in use'''
#pert_start = -10
#pert_end = 11
#pert_diff = 1

#Perturb variable matrix
#first item in each list is start, second is end, last is increment
#first list in pert list is dip, then dip direct and pitch
pert_1st = [[-10,11,1],[-10,11,1],[-10,11,1]]
pert_2nd = [[-10,11,1],[-10,11,1],[-10,11,1]]
pert = [pert_1st,pert_2nd]

#fold_type = 1 for single folds history file, = 2 for non-single fold 
fold_type = 2

#number of history files going to generate 
num_hist = 100

#number of events
num_event = 2

#interference type
#set to 0 for single generation
#type 0 to 3 for two generation
interference = 3

#number of fold in each event
#should automatically change my the program
num_fold = [1,3]

#directories to save
single_1gen = "1st_gen_single"
nonSingle_1gen = "1st_gen_Nsingle"
single_2gen = "2nd_gen_single"
nonSingle_2gen_type0 = "2nd_gen_Nsingle_type0"
nonSingle_2gen_type1 = "2nd_gen_Nsingle_type1"
nonSingle_2gen_type2 = "2nd_gen_Nsingle_type2"
nonSingle_2gen_type3 = "2nd_gen_Nsingle_type3"
single_2gen_type2 = "2nd_gen_type2"

#Original history file name

#Events properties store in list
#first value for the properties of first event
dip = [90, 90]
dip_dir = [90, 180]
pitch = [0, 0]

#Use defined values
#False for using the value in history file
'''WARNING : THIS WILL CHANGE THE ORIGIN HISTORY FILE'''
use_value = False


#initialize random object for each perturbation
for n in xrange(num_event):
    rnd_list.append(random)
    rnd_list[n].seed(rnd_seed_list[n])


# Define fold generation properties
def prop(history,num_event,num_fold,dip,dip_dir,pitch):
    #for each generation, initialize with pre-defined variable
    for event in xrange(num_event):
        start = 2+sum(num_fold[0:event:1])
        for fold in range(start,start+num_fold[event],1):
            history.events[fold].properties['Dip'] = dip[event]
            history.events[fold].properties['Dip Direction'] = dip_dir[event]
            history.events[fold].properties['Pitch'] = pitch[event]
    return history
    
# Perturb Fold

# In[ ]:

def perturb(history,start_event,num_fold,num_event,hist_no):
    
    for event in xrange(num_event):
        start = 2+sum(num_fold[0:event:1])
        for fold in range(start,start+num_fold[event],1):
            dip_change = rnd_list[event].randrange(pert[event][0][0],pert[event][0][1],pert[event][0][2])
            dip_dir_change = rnd_list[event].randrange(pert[event][1][0],pert[event][1][1],pert[event][1][2])
            pitch_change = rnd_list[event].randrange(pert[event][2][0],pert[event][2][1],pert[event][2][2])
            history.events[fold].properties['Dip'] += dip_change
            history.events[fold].properties['Dip Direction'] += dip_dir_change
            history.events[fold].properties['Pitch'] += pitch_change
            fold_change = [dip_change, dip_dir_change, pitch_change]
            change_log[hist_no,fold-2,:] = fold_change
    return history

# Prepare file directory for non-single fold or single fold 


# In[ ]:
if (fold_type == 1 and num_event == 1 and interference == 0):
    save_path = single_1gen
    history_file  =  "init_1gen.his"
elif (fold_type == 2 and num_event == 1 and interference == 0):
    save_path = nonSingle_1gen
    history_file  =  "init_1gen_ns.his"
elif (fold_type == 1 and num_event == 2 and interference == 1):
    save_path = single_2gen
    history_file  =  "init_2gen.his"
elif (fold_type == 2 and num_event == 2 and interference == 0):
    save_path = nonSingle_2gen_type0
    history_file  =  "init_2gen_type0_ns.his"
elif (fold_type == 2 and num_event == 2 and interference == 1):
    save_path = nonSingle_2gen_type1
    history_file  =  "init_2gen_type1_ns.his"
elif (fold_type == 2 and num_event == 2 and interference == 2):
    save_path = nonSingle_2gen_type2
    history_file  =  "init_2gen_type2_ns.his"
elif (fold_type == 2 and num_event == 2 and interference == 3):
    save_path = nonSingle_2gen_type3
    history_file  =  "init_2gen_type3_ns.his"
elif (fold_type == 1 and num_event == 2 and interference == 2):
    save_path = single_2gen_type2
    history_file  =  "init_2gen_type2.his"
else:
    save_path = nonSingle_2gen
    history_file  =  "init_2gen_ns.his"
    
if (fold_type == 2):
    num_fold = [1,1]

#changes of each model
change_log = np.ndarray(shape=(num_hist,sum(num_fold[0:num_event]),3), dtype = float, order = 'F')
    
if not os.path.exists(os.path.join(repo_path, save_path)):
    os.makedirs(os.path.join(repo_path,save_path))
os.chdir(os.path.join(repo_path,save_path))

# Actual Perturbing task

# In[ ]:

history_path  =  os.path.join(repo_path,  history_file)

history = pynoddy.NoddyHistory(history_path)

if (use_value):
    history = prop(history,num_event,num_fold,dip,dip_dir,pitch)
    history.write_history(history_path)
    history = pynoddy.NoddyHistory(history_path)

print "Start Perturbing"

for gen_hist in range(1,num_hist+1,1):
    history = perturb(history,2,num_fold,num_event,gen_hist-1)
    perturbed_path = str(gen_hist)+'.his'
    history.write_history(perturbed_path)
    history = pynoddy.NoddyHistory(history_path);
    
print "Start Generating output and diagram"
for load_hist in range(1,num_hist+1,1):
    if load_hist%10 == 0:
        print "Processing history file " + str(load_hist)
    perturbed_path = str(load_hist)+'.his'
    output_name = "out_"+str(load_hist)
    pynoddy.compute_model(perturbed_path,output_name)
    model_output = pynoddy.output.NoddyOutput(output_name)
    model_output.export_to_vtk(vtk_filename = output_name)
    
    #Figure generate
    fig  =  plt.figure(figsize  =  (15,5))
    ax1  =  fig.add_subplot(131)
    ax2  =  fig.add_subplot(132)
    ax3  =  fig.add_subplot(133)
    model_output.plot_section('x',  position='center',  ax  =  ax1,  colorbar=False,  title="Cross-section of X-axis")
    model_output.plot_section('y',  position='center',  ax  =  ax2,  colorbar=False,  title="Cross-section of Y-axis")
    model_output.plot_section('z',  position='center',  ax  =  ax3,  colorbar=False,  title="Cross-section of Z-axis")
    plt.savefig(output_name+"fig")
    #reset to init
    reload(pynoddy.history)
    reload(pynoddy.output)
    plt.close()
    
#Save result
with file('fold_parameter_change.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(change_log.shape))
    outfile.write('# (Models, Folds, Fold parameters)\n')
    for data_slice in change_log:
  
        np.savetxt(outfile, data_slice)

        outfile.write('# Next Model\n')
        
print "Completed"

