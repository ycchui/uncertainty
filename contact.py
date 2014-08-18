import pynoddy
import os
import numpy as np
import vtk
import time

#directory of model suite
store_path = "2nd_gen_Nsingle_type3"

#number of model in the suite
num_model = 100

#number of lithology
lith = 7

#load path
repo_path  =  os.path.realpath("")
os.chdir(os.path.join(repo_path,store_path))

def save_cur(name, result):
    with file(name+".txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(result.shape))
        outfile.write('# File format: contact1/2(gauss), contact1/2(mean), contact2/3(gauss), contact2/3(mean),...\n')
        np.savetxt(outfile, result)
def save_area(name, result):
    with file(name+".txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(result.shape))
        outfile.write('# File format: contact1/2, contact2/3, contact3/4,...\n')
        np.savetxt(outfile, result)

#read first model
model_name = "out_1"
model_output = pynoddy.output.NoddyOutput(model_name)
model_suite = model_output.block.astype(int)
model_suite = model_suite[None,...]

#get length of model
x_len = model_output.nx
y_len = model_output.ny
z_len = model_output.nz

print "READING MODELS"
'''#read model
for load_model in range(2, num_model+1, 1):
    model_name = "out_"+str(load_model)
    model_output = pynoddy.output.NoddyOutput(model_name)
    model_suite = np.append(model_suite,model_output.block[None,...].astype(int), axis = 0)'''

#number of contact = lith-1 if all lithologies are continuous
model_contact_area = np.zeros(lith-1)

#dataImporter = vtk.vtkImageImport()
dataReader = vtk.vtkXMLImageDataReader()
for model in range(0, num_model, 1):
    #read vti file
    model_name = "out_"+str(model+1)+".vti"
    dataReader.SetFileName(model_name)
    dataReader.Update()
    pd = dataReader.GetOutput().GetPointData().GetArray(0)
    input = dataReader.GetOutput()
    input.GetPointData().SetScalars(pd)
    #MARCHING CUBES
    model_area = np.zeros(1)
    model_sum_cur_g = np.zeros(1)
    model_sum_cur_g_std = np.zeros(1)
    model_avg_cur_g = np.zeros(1)
    model_avg_cur_g_std = np.zeros(1)
    model_sum_cur_m = np.zeros(1)
    model_sum_cur_m_std = np.zeros(1)
    model_avg_cur_m = np.zeros(1)
    model_avg_cur_m_std = np.zeros(1)
    for bound in range(1, lith, 1):
        #contacts = vtk.vtkDiscreteMarchingCubes()
        #contacts.SetInputData(input)
        contacts = vtk.vtkMarchingCubes()
        contacts.SetInputData(input)
        true_bound = bound+0.5
        contacts.SetValue(0,true_bound)
        contacts.Update()

        surface_area = vtk.vtkMassProperties()
        surface_area.SetInputConnection(contacts.GetOutputPort())
        surface_area.Update()

        curvature = vtk.vtkCurvatures()
        curvature.SetInputConnection(contacts.GetOutputPort())
        curvature.SetCurvatureTypeToMinimum()
        #curvature.SetCurvatureTypeToMaximum()
        #curvature.SetCurvatureTypeToGaussian()
        #curvature.SetCurvatureTypeToMean()
        curvature.Update()
        gauss = np.frombuffer(curvature.GetOutput().GetPointData().GetArray("Gauss_Curvature"), dtype = float)
        mean = np.frombuffer(curvature.GetOutput().GetPointData().GetArray("Mean_Curvature"), dtype = float)
        if (true_bound ==1.5):
            model_area[0] = surface_area.GetSurfaceArea()
            model_sum_cur_g[0] = np.sum(gauss)
            model_sum_cur_m[0] = np.sum(mean)
            model_sum_cur_g_std[0] = np.std(gauss)
            model_sum_cur_m_std[0] = np.std(mean)
            model_avg_cur_g[0] = np.mean(gauss)
            model_avg_cur_m[0] = np.mean(mean)
            model_avg_cur_g_std[0] = np.std(gauss)
            model_avg_cur_m_std[0] = np.std(mean)
        else:
            model_area = np.hstack((model_area,surface_area.GetSurfaceArea()))
            model_sum_cur_g = np.hstack((model_sum_cur_g,np.sum(gauss)))
            model_avg_cur_g = np.hstack((model_avg_cur_g,np.mean(gauss)))
            model_sum_cur_g_std = np.hstack((model_sum_cur_g_std,np.std(gauss)))
            model_avg_cur_g_std = np.hstack((model_avg_cur_g_std,np.std(gauss)))
            model_sum_cur_m = np.hstack((model_sum_cur_m,np.sum(mean)))
            model_avg_cur_m = np.hstack((model_avg_cur_m,np.mean(mean)))
            model_sum_cur_m_std = np.hstack((model_sum_cur_m_std,np.std(mean)))
            model_avg_cur_m_std = np.hstack((model_avg_cur_m_std,np.std(mean)))
    if (model==0):
        suite_area = model_area
        suite_sum_cur_g = model_sum_cur_g
        suite_avg_cur_g = model_avg_cur_g
        suite_sum_cur_g_std = model_sum_cur_g_std
        suite_avg_cur_g_std = model_avg_cur_g_std
        suite_sum_cur_m = model_sum_cur_m
        suite_avg_cur_m = model_avg_cur_m
        suite_sum_cur_m_std = model_sum_cur_m_std
        suite_avg_cur_m_std = model_avg_cur_m_std
    else:
        suite_area = np.vstack((suite_area,model_area))
        suite_sum_cur_g = np.vstack((suite_sum_cur_g,model_sum_cur_g))
        suite_avg_cur_g = np.vstack((suite_avg_cur_g,model_avg_cur_g))
        suite_sum_cur_g_std = np.vstack((suite_sum_cur_g_std,model_sum_cur_g_std))
        suite_avg_cur_g_std = np.vstack((suite_avg_cur_g_std,model_avg_cur_g_std))
        suite_sum_cur_m = np.vstack((suite_sum_cur_m,model_sum_cur_m))
        suite_avg_cur_m = np.vstack((suite_avg_cur_m,model_avg_cur_m))
        suite_sum_cur_m_std = np.vstack((suite_sum_cur_m_std,model_sum_cur_m_std))
        suite_avg_cur_m_std = np.vstack((suite_avg_cur_m_std,model_avg_cur_m_std))
    if model%10 == 0:
        print "Computing Model " + str(model)

#print dataReader.GetOutput()
#print surface_area.GetSurfaceArea()

#print data.shape

#print contacts.GetOutput()
'''for test in range(0,10,1):
    print result.GetCell(test)'''
#generate vti file from result
'''writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('test_contact.vtp')
writer.SetInputData(result)
writer.Write()'''

#generate vti file from input
'''writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName('test_contact.vti')
writer.SetInputData(dataReader.GetOutput())
writer.Write()'''

save_area('contact_surface_area',suite_area)
save_cur('contact_curvature_sum_gauss',suite_sum_cur_g)
save_cur('contact_curvature_avg_gauss',suite_avg_cur_g)
save_cur('contact_curvature_sum_std_gauss',suite_sum_cur_g_std)
save_cur('contact_curvature_avg_std_gauss',suite_avg_cur_g_std)
save_cur('contact_curvature_sum_mean',suite_sum_cur_m)
save_cur('contact_curvature_avg_mean',suite_avg_cur_m)
save_cur('contact_curvature_sum_std_mean',suite_sum_cur_m_std)
save_cur('contact_curvature_avg_std_mean',suite_avg_cur_m_std)
