import pynoddy
import os
import numpy as np
import matplotlib.pyplot as plt

#directory of model suite
store_path = "2nd_gen_Nsingle_type3"

#number of model in the suite
num_model = 100

#number of lithology
lith = 7

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))

#read first model
model_name = "out_1"
model_output = pynoddy.output.NoddyOutput(model_name)
block_model = model_output.block.astype(int)
#model_suite = model_output.block.astype(int)
#model_suite = model_suite[None,...]

#get length of model
x_len = model_output.nx
y_len = model_output.ny
z_len = model_output.nz

#Calculate neighbourhood
def cal_neighbour(model, x, y, z):
    #local = model[x,y,z]
    temp = np.zeros(6)
    if (x>0):
        temp[0] = model[x-1,y,z]
    if (x<x_len-1):
        temp[1] = model[x+1,y,z]
    if (y>0):
        temp[2] = model[x,y-1,z]
    if (y<y_len-1):
        temp[3] = model[x,y+1,z]
    if (z>0):
        temp[4] = model[x,y,z-1]
    if (z<z_len-1):
        temp[5] = model[x,y,z+1]
    return model[x,y,z], np.count_nonzero(np.unique(temp))

def gen_fig(result, x_len, y_len, z_len, fig_title):
    fig  =  plt.figure(figsize  =  (15,5))
    ax1  =  fig.add_subplot(131)
    ax2  =  fig.add_subplot(132)
    ax3  =  fig.add_subplot(133)
    #x-axis
    ax = ax1
    cell_pos = x_len / 2
    section_slice = result[cell_pos,:,:].transpose()
    xlabel = "y"
    ylabel = "z"
    title = fig_title + " from x-axis"
    im = ax.imshow(section_slice, interpolation='nearest', aspect=2., cmap='jet', origin = 'lower left')
    cbar = plt.colorbar(im)
    _ = cbar
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #y-axis
    ax = ax2
    cell_pos = y_len / 2
    section_slice = result[:,cell_pos,:].transpose()
    xlabel = "x"
    ylabel = "z"
    title = fig_title +" from y-axis"
    im = ax.imshow(section_slice, interpolation='nearest', aspect=2., cmap='jet', origin = 'lower left')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #z-axis
    ax = ax3
    cell_pos = z_len / 2
    section_slice = result[:,:,cell_pos].transpose()
    xlabel = "x"
    ylabel = "y"
    title = fig_title+" from z-axis"
    im = ax.imshow(section_slice, interpolation='nearest', aspect=1., cmap='jet', origin = 'lower left')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(fig_title)
    
#Save result
def save_result(name, result):
    with file(name+".txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(result.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in result:  
            np.savetxt(outfile, data_slice)

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
model_lith = np.zeros(lith)
#lith_count = np.zeros(lith)
lith_count = np.delete(np.bincount(block_model.ravel()),0, None)
#compute neighbourhood of first model
for x in range(0, x_len, 1):
        for y in range(0, y_len, 1):
            for z in range(0, z_len, 1):
                #lith_count[block_model[x,y,z]-1]+=1
                cur_lith, no_neig = cal_neighbour(block_model,x,y,z)
                model_lith[cur_lith-1] += no_neig
                #model_lith[block_model[x,y,z]-1]+=cal_neighbour(block_model,x,y,z)
                
lith_neighbour = model_lith/lith_count
model_lith.fill(0)
lith_count.fill(0)

print "READING MODELS"
#read model
for load_model in range(2, num_model+1, 1):
    print "READING MODEL", load_model
    model_name = "out_"+str(load_model)
    model_output = pynoddy.output.NoddyOutput(model_name)
    block_model = model_output.block.astype(int)
    
    lith_count = np.delete(np.bincount(block_model.ravel()),0, None)
    print "COMPUTING MODEL", load_model
    #computing lithology neighbour
    for x in range(0, x_len, 1):
        for y in range(0, y_len, 1):
            for z in range(0, z_len, 1):
                #lith_count[block_model[x,y,z]-1]+=1
                #model_lith[block_model[x,y,z]-1]+=cal_neighbour(block_model,x,y,z)
                cur_lith, no_neig = cal_neighbour(block_model,x,y,z)
                model_lith[cur_lith-1] += no_neig
                
    lith_neighbour = np.vstack((lith_neighbour, model_lith/lith_count))
    model_lith.fill(0)
    lith_count.fill(0)

    #model_suite = np.append(model_suite,model_output.block[None,...].astype(int), axis = 0)

#result = np.ndarray(shape=(num_model,x_len,y_len,z_len), dtype = int, order = 'F')

#The following block of code compute neighbourhood and lithology average separately.
#This require >2GB of memory in a model with large amount of cubes, which could cause
#the script crash.
'''print "COMPUTE NEIGHBOURHOOD"
for model in range(0, num_model, 1):
    for x in range(0, x_len, 1):
        for y in range(0, y_len, 1):
            for z in range(0, z_len, 1):
                result[model,x,y,z] = cal_neighbour(model_suite[model,:,:,:],x,y,z)
    print "COMPUTING MODEL", model+1
    #save_result("neighbourhood_"+model+1,result[model,:,:,:]
print "COMPUTE LITHOLOGY AVERAGE"

for model in range(0, num_model, 1):
    for x in range(0, x_len, 1):
        for y in range(0, y_len, 1):
            for z in range(0, z_len, 1):
                lith_count[model_suite[model,x,y,z]-1]+=1
                model_lith[model_suite[model,x,y,z]-1]+=result[model,x,y,z]
    if model == 0 :
        lith_neighbour = model_lith/lith_count
    else:
        lith_neighbour = np.vstack((lith_neighbour, model_lith/lith_count))
    model_lith.fill(0)
    lith_count.fill(0)'''
#to check possible error, might takes multiple hours
#check = np.argmax(result>6)

#save neighbourhood relationship
'''with file("neighbourhood.txt", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(result.shape))
    for model_slice in result:
        for data_slice in model_slice:
            np.savetxt(outfile, data_slice)
            outfile.write('# New slice\n')
        outfile.write('# New model\n')'''


'''suite_avg = np.ndarray(shape=(x_len,y_len,z_len), dtype = float, order = 'F')

print "COMPUTE MODEL SUITE AVERAGE"
for x in range(0, x_len, 1):
    for y in range(0, y_len, 1):
        for z in range(0, z_len, 1):
            suite_avg[x,y,z] = np.mean(result, axis = 0)'''
#suite_avg = np.mean(result,axis=0)
            
#print suite_avg.shape
#gen_fig(suite_avg,x_len,y_len,z_len,"Model Suite Average Neighbourhood")
#save_result('model_suite_avg_neighbour',suite_avg)


with file("unit_neighbour.txt", 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(lith_neighbour.shape))
    outfile.write('# File format: unit1, unit2, unit3, ...\n')
    np.savetxt(outfile, lith_neighbour)