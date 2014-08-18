import pynoddy
import os
import numpy as np

#directory of model suite
store_path = "2nd_gen_Nsingle_type3"

#number of model in the suite
num_model = 100

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))

#read first model
model_name = "out_1"
model_output = pynoddy.output.NoddyOutput(model_name)
model_suite = model_output.block.astype(int)
model_suite = model_suite[None,...]

#get length of model
x_len = model_output.nx
y_len = model_output.ny
z_len = model_output.nz

#Find deepest of each unit
def find_deep(model, num_unit):
    unit_done = 0
    deep = np.zeros(num_unit)
    deep.fill(z_len)
    layer = 0
    while (unit_done<num_unit and layer<z_len):
        temp = np.unique(model[:,:,layer])
        for occurence in temp:
            #index of unit start from zero, while unit in model start from 1
            if (deep[occurence-1] > layer):
                deep[occurence-1] = layer
                unit_done += 1
        layer += 1
        
    return deep
#Find shallowest of each unit
def find_shallow(model, num_unit):
    unit_done = 0
    shallow = np.zeros(num_unit)
    shallow.fill(-1)
    layer = z_len-1
    while (unit_done<num_unit and layer>-1):
        temp = np.unique(model[:,:,layer])
        for occurence in temp:
            #index of unit start from zero, while unit in model start from 1
            if (shallow[occurence-1] < layer):
                shallow[occurence-1] = layer
                unit_done += 1
        layer -= 1
    return shallow
        
#Save result
def save_result(name, result):
    with file(name+".txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(result.shape))
        outfile.write('# File format: unit1, unit2, unit3, ...\n')
        np.savetxt(outfile, result)

result_vol = np.delete(np.bincount(model_suite[0,:,:,:].ravel()),0, None)
num_unit = result_vol.size
result_deep = find_deep(model_suite[0,:,:,:],num_unit)
result_shallow = find_shallow(model_suite[0,:,:,:],num_unit)

print "COMPUTING DEPTH AND VOLUME"
#read model
for load_model in range(2, num_model+1, 1):
    model_name = "out_"+str(load_model)
    model_output = pynoddy.output.NoddyOutput(model_name)
    #model_suite = np.append(model_suite,model_output.block[None,...].astype(int), axis = 0)
    block_model = model_output.block.astype(int)

#compute depth and volume

    #result_vol = np.vstack((result_vol,np.delete(np.bincount(model_suite[model,:,:,:].ravel()),0,None)))
    #result_deep = np.vstack((result_deep,find_deep(model_suite[model,:,:,:],num_unit)))
    result_vol = np.vstack((result_vol,np.delete(np.bincount(block_model.ravel()),0,None)))
    #result_shallow = np.vstack((result_shallow,find_shallow(model_suite[model,:,:,:],num_unit)))
    result_deep = np.vstack((result_deep,find_deep(block_model,num_unit)))
    result_shallow = np.vstack((result_shallow,find_shallow(block_model,num_unit)))
    if load_model%10 == 0:
        print "Computing Model " + str(load_model)

save_result('unit_volume', result_vol)
save_result('unit_deepest', result_deep)
save_result('unit_shallowest', result_shallow)