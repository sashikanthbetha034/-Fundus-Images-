import os
import h5py
import cv2 as cv
import pandas as pd
from SOA import SOA
from LEA import LEA
from POA import POA
from OOA import OOA
from MLEA import MLEA
from numpy import matlib
from Plot_Results import *
from Model_ANN import Model_ANN
from Model_CNN import Model_CNN
from Obj_Fun import Obj_fun_CLS
from Model_FCN import Model_FCN
from Models_Ran import Model_Ran
from Global_Vars import Global_Vars
from Model_TLACNN import Model_TLACNN
from Model_Densenet import Model_Dil_Densenet


def Read_Image(filename):
    image = cv.imread(filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


def Read_Dataset():
    Images = []
    # Diabetic Retinopathy (Read image)
    Directory = './Dataset/Imagenes/'
    path = './Dataset/idrid_labels.csv'
    my_csv = np.asarray(pd.read_csv(path))
    value = my_csv[:, 1]
    value2 = my_csv[:, 0]
    List_dir = os.listdir(Directory)
    for j in range(len(List_dir)):
        file = value2[j] + '.jpg'
        filename = Directory + file
        Images.append(Read_Image(filename))

    # Diabetic Macular Edema (Read image)
    Directory2 = './Dataset/messidor-2/messidor-2/preprocess/'
    path2 = './Dataset/messidor_data.csv'
    my_csv2 = np.asarray(pd.read_csv(path2))
    Target_Column = my_csv2[:, 2]
    Data_Column = my_csv2[:, 0]
    Target_Class = []
    for k in range(len(Target_Column)):
        if Target_Column[k] == 1:
            Target_Class.append(np.max((value) + 1))
            filenames = Directory2 + Data_Column[k]
            Images.append(Read_Image(filenames))

    # Concatinating Targets
    Target = np.zeros((len(value)) + len(Target_Class)).astype('int')
    Target[:len(value)] = value
    Target[len(value):] = Target_Class

    uniq = np.unique(Target)
    Targ = np.zeros((len(Target), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(Target == uniq[i])
        Targ[index, i] = 1

    # Shuffle Data and Target
    Image = np.array(Images)
    indices = np.arange(Image.shape[0])
    np.random.shuffle(indices)
    shuffled_data = (Image[indices])[: -1, :, :]
    shuffled_target = Targ[indices][: -1, :]

    return shuffled_data, shuffled_target


# Read Dataset
an = 0
if an == 1:
    Images, Target = Read_Dataset()
    np.save('Images.npy', Images)
    np.save('Target.npy', Target)

# Get weights from RAN and Multi-Dilated Densenet
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    mod_1 = Model_Ran(Images, Target)
    mod_2 = Model_Dil_Densenet(Images, Target)
    with h5py.File('Model_RAN.h5', 'w') as f:
        f.create_dataset("RAN", data=mod_1)
    with h5py.File('Model_Dil_Densenet.h5', 'w') as f:
        f.create_dataset("Dense", data=mod_2)

# Optimization for Data Classification
an = 0
if an == 1:
    with h5py.File('Model_RAN.h5', 'r') as file:
        weight_1 = file['RAN'][:]
    with h5py.File('Model_Dil_Densenet.h5', 'r') as f:
        weight_2 = f['Dense'][:]
    RAN_weight = weight_1
    Dense_weight = weight_2

    Data = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Hidden_neuron_count = 2
    Global_Vars.Data = Data
    Global_Vars.RAN_weight = RAN_weight
    Global_Vars.Dense_weight = Dense_weight
    Global_Vars.Target = Target
    Npop = 10
    Chlen = Hidden_neuron_count
    xmin = matlib.repmat([RAN_weight.shape[1] * (-0.2), Dense_weight.shape[1] * (-0.2)], Npop, 1)
    xmax = matlib.repmat([RAN_weight.shape[1] * 0.2, Dense_weight.shape[1] * 0.2], Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = Obj_fun_CLS
    Max_iter = 50

    print("MAO...")
    [bestfit1, fitness1, bestsol1, time1] = OOA(initsol, fname, xmin, xmax, Max_iter)  # Osprey Optimization Algorithm (OOA)

    print("TSO...")
    [bestfit2, fitness2, bestsol2, time2] = POA(initsol, fname, xmin, xmax, Max_iter)  # Pufferfish Optimization Algorithm (POA)

    print("BWO...")
    [bestfit3, fitness3, bestsol3, time3] = SOA(initsol, fname, xmin, xmax, Max_iter)  # Sculptor Optimization Algorithm (SOA)

    print("LEA...")
    [bestfit4, fitness4, bestsol4, time4] = LEA(initsol, fname, xmin, xmax, Max_iter)  # Lotus Effect Optimization Algorithm

    print("MLEA..")
    [bestfit5, fitness5, bestsol5, time5] = MLEA(initsol, fname, xmin, xmax, Max_iter)  # Modified Lotus Effect Optimization Algorithm

    sols = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    Fit = [fitness1, fitness2, fitness3, fitness4, fitness5]
    np.save('Fitness.npy', Fit)
    np.save('bestsol.npy', sols)

# Classification
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    Bestsol = np.load('bestsol.npy', allow_pickle=True)  # loading step

    with h5py.File('Model_RAN.h5', 'r') as file:
        weight_1 = file['RAN'][:]
    with h5py.File('Model_Dil_Densenet.h5', 'r') as f:
        weight_2 = f['Dense'][:]
    RAN_weight = weight_1
    Dense_weight = weight_2

    Act = []
    Batch_size = [16, 32, 64, 128, 256]
    for act in range(len(Batch_size)):
        learnperc = round(Images.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Images[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Images[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(Bestsol.shape[0]):
            print(act, j)
            sol = np.round(Bestsol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_TLACNN(RAN_weight, Dense_weight, Test_Data, Test_Target, Batch_size[act], sol)  # Model GRU
        Eval[5, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])  # Model GRU
        Eval[6, :], pred1 = Model_FCN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])  # Model BiLSTM
        Eval[7, :], pred2 = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target, Batch_size[act])  # Model LSTM
        Eval[8, :], pred3 = Model_TLACNN(RAN_weight, Dense_weight, Test_Data, Test_Target, Batch_size[act])  # Model DTCN
        Eval[9, :], pred4 = Eval[4, :]
        Act.append(Eval)
    np.save('Eval_all.npy', Act)  # Save Eval all

Plot_ROC_Curve()
plotConvResults()
plot_results_KFOLD_Positive_Measures()
Plot_Results_Batch()
Plot_Confusion()
