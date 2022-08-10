import scipy.io
import numpy as np
import matplotlib.pyplot as plt

folder_path = '/home/kamo/resources/tip2014/'

# Constants in Figen's code
lambda0 = 195.12
LightSpeed = 300000
Dispersion = 50

amo = scipy.io.loadmat(folder_path + 'realData1_3.mat')
inten = amo['intensity']
vel = amo['velocity'] * lambda0*1000/LightSpeed / Dispersion
width = amo['width'] * 1000 / Dispersion

meas_0 = amo['dispersedI0']
meas_1 = amo['dispersedI1']
meas_m1 = amo['dispersedI2']

est_inten = amo['estF']
est_vel = amo['estD']
est_width = amo['estW']