import subprocess
import numpy as np 
import pandas as pd
import os
import opensim as osim
from ezc3d import c3d
from stan_utils import *


def runProgram(argList):
# arglist is like ['./printNumbers.sh']
    proc = subprocess.Popen(argList, 
                            shell=False, bufsize=1, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT)
    while (True):
        # Read line from stdout, print, break if EOF reached
        line = proc.stdout.readline()
        line = line.decode()
        if (line == ""): break
        print(line),
        
    rc = proc.poll() 
    print('\nReturn code: ', rc, '\n')
    return rc

def getField( txtLog, strFieldName ):
    idx1 = txtLog.index(strFieldName) + len(strFieldName)
    idx2 = txtLog.index('\n', idx1)
    strField= txtLog[idx1:idx2]
    return strField


def getMassOfModel(osimModel):
    totalMass = 0
    allBodies = osimModel.getBodySet()
    for i in range(0, allBodies.getSize()):
        curBody = allBodies.get(i)
        totalMass = totalMass + curBody.getMass()
    return totalMass


def setMassOfBodiesUsingRRAMassChange(osimModel, massChange):
    currTotalMass = getMassOfModel(osimModel)
    suggestedNewTotalMass = currTotalMass + massChange
    massScaleFactor = suggestedNewTotalMass/currTotalMass
    allBodies = osimModel.getBodySet()
    for i in range(0, allBodies.getSize()):
        curBody = allBodies.get(i)
        currBodyMass=curBody.getMass()
        newBodyMass = currBodyMass*massScaleFactor
        curBody.setMass(newBodyMass)
        
    return osimModel    


def scaleOptimalForceSubjectSpecific(osimModel_generic, osimModel_scaled, height_generic, height_scaled):
    mass_generic = getMassOfModel(osimModel_generic)
    mass_scaled = getMassOfModel(osimModel_scaled)
    
    Vtotal_generic = 47.05 * mass_generic * height_generic + 1289.6
    Vtotal_scaled = 47.05 * mass_scaled * height_scaled + 1289.6
    
    allMuscles_generic = osimModel_generic.getMuscles()
    allMuscles_scaled = osimModel_scaled.getMuscles()
    for i in range(0, allMuscles_generic.getSize()):
        currentMuscle_generic = allMuscles_generic.get(i)
        currentMuscle_scaled = allMuscles_scaled.get(i)
        
        lmo_generic = currentMuscle_generic.getOptimalFiberLength()
        lmo_scaled = currentMuscle_scaled.getOptimalFiberLength()

        forceScaleFactor = (Vtotal_scaled/Vtotal_generic)/(lmo_scaled/lmo_generic)
        
        currentMuscle_scaled.setMaxIsometricForce( forceScaleFactor * currentMuscle_generic.getMaxIsometricForce() );

    return osimModel_scaled
        

def setMaxContractionVelocityAllMuscles(osimModel, maxContractionVelocity):
    Muscles = osimModel.getMuscles()
    
    for i in range(0, Muscles.getSize()):
        currentMuscle = Muscles.get(i)
        currentMuscle.setMaxContractionVelocity(maxContractionVelocity)
    
    return osimModel


def matRMS(vals):
    rms = np.sqrt(np.mean(vals**2, axis = 0))
    return rms


# Export EMG data to a csv file
def importEMG(input_dir, file_name, output_dir):
    filename = os.path.join(input_dir, file_name)
    c = c3d(filename)

    analogs = c['data']['analogs']
    data=analogs[0]
    labels_list = c['parameters']['ANALOG']['LABELS']['value']

    emg_data = []
    emg_list=[]
    for i, l in enumerate(labels_list):
        if 'Voltage' in l:
            emg_list.append(l[8:])
            emg_data.append(data[i])

    emg_df = pd.DataFrame()        
    for i, label in enumerate(emg_list):
        signal = calculate_emg_linear_envelope(emg_data[i])
        emg_df[label] = signal.tolist()
    to_write = os.path.join(output_dir, 'emg.csv')
    emg_df.to_csv(to_write, index=False)
    return emg_df