import pynoddy
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#directory of model suite
store_path = "2nd_gen_type2"

#number of model in the suite
num_model = 100

#numpy array to store model suite
#model_suite = 

#Generate 2D figure
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

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))

#read first model
model_name = "out_1"
model_output = pynoddy.output.NoddyOutput(model_name)
model_suite = model_output.block.copy()
model_suite = model_suite[None,...]

#get length of model
x_len = model_output.nx
y_len = model_output.ny
z_len = model_output.nz

print "READING MODELS"
#read model
for load_model in range(2, num_model+1, 1):
    model_name = "out_"+str(load_model)
    model_output = pynoddy.output.NoddyOutput(model_name)
    model_suite = np.append(model_suite,model_output.block[None,...], axis = 0)

result = np.ndarray((x_len,y_len,z_len))
#number of appearance of single unit
result_layer = np.ndarray((x_len,y_len,z_len))

print "COMPUTING INFORMATION ENTROPY"
for x in range(0, x_len, 1):
    for y in range(0, y_len, 1):
        for z in range(0, z_len, 1):
            #uind is an array of the index of elements in original array in new array (ua)
            ua, uind = np.unique(model_suite[:,x,y,z], return_inverse = True)
            #count the repeat of each element in uind to find the appearance of each element in the model suite
            count = np.bincount(uind)
            cell_entropy = 0
            #finding entropy of each cell, using the formula "summation of p*log(p)" (Wellmann, 2011)
            for occurence in count:
                prob = float(occurence)/float(num_model)
                cell_entropy += (prob)*math.log(prob)
            result[x,y,z] = -cell_entropy
            if 4 in ua:
                result_layer[x,y,z] = count[np.nonzero(ua == 4)[0][0]]
            
print result

print "GENERATE FIGURE"
gen_fig(result, x_len, y_len, z_len, 'Information Entropy')
gen_fig(result_layer, x_len, y_len, z_len, 'Occurrence of unit 4')
'''#Generate 2D figure
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
title = "Information Entropy from x-axis"
im = ax.imshow(section_slice, interpolation='nearest', aspect=1., cmap='jet', origin = 'lower left')
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
title = "Information Entropy from y-axis"
im = ax.imshow(section_slice, interpolation='nearest', aspect=1., cmap='jet', origin = 'lower left')
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
#z-axis
ax = ax3
cell_pos = z_len / 2
section_slice = result[:,:,cell_pos].transpose()
xlabel = "x"
ylabel = "y"
title = "Information Entropy from z-axis"
im = ax.imshow(section_slice, interpolation='nearest', aspect=1., cmap='jet', origin = 'lower left')
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
plt.savefig("Information Entropy")'''

#Save result
with file('test.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(result.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in result:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice)

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')

