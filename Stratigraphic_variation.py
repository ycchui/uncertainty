import pynoddy
import os
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#directory of model suite
store_path = "2nd_gen_type2"

#number of model in the suite
num_model = 100

#numpy array to store model suite
#model_suite = 

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

print "READ MODELS"
#read model
for load_model in range(2, num_model+1, 1):
    model_name = "out_"+str(load_model)
    model_output = pynoddy.output.NoddyOutput(model_name)
    model_suite = np.append(model_suite,model_output.block[None,...], axis = 0)
    '''if load_model == 2:
        #print model_output.block
        print x_len
        print y_len
        print z_len
        print model_suite[0, 0, :, 24]
print model_suite.shape'''

result = np.ndarray((x_len,y_len,z_len))
result_variability = np.ndarray((x_len,y_len,z_len))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')

pt_count = 0

print "START COMPUTING VARIATION"
#compute cell variation
for x in range(0, x_len, 1):
    for y in range(0, y_len, 1):
        for z in range(0, z_len, 1):
            ua, uind = np.unique(model_suite[:,x,y,z], return_inverse = True)
            result[x,y,z] = np.count_nonzero(ua)
            result_variability[x,y,z] = float(num_model - np.amax(np.bincount(uind)))/float(num_model)
            #if result[x,y,z] > 1:
                #ax.scatter(x,y,z,c=result[x,y,z])
            #pt_count +=1
            #print pt_count
print result

print "GENERATE FIGURE"
gen_fig(result, x_len,y_len,z_len, "Stratigraphic possibility")
gen_fig(result_variability, x_len,y_len,z_len, "Stratigraphic variability")
save_result("Stratigraphic possibility", result)
save_result("Stratigraphic variability", result_variability)



'''print "GENERATE DIAGRAM"
for x in range(0, x_len, 1):
    for y in range(0, y_len, 1):
        for z in range(0, z_len, 1):
            if result[x,y,z] > 1:
                if result[x,y,z] == 1:
                    color = 'r'
                elif result[x,y,z] == 2:
                    color = 'b'
                else:
                    color = 'g'
                ax.scatter(x,y,z,c=color)
            pt_count+=1
            if pt_count%100 == 0:
                print pt_count
plt.savefig("cell_var.png")
plt.show()'''
