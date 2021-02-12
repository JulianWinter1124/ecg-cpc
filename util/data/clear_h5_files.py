import glob
import os

if __name__ == '__main__':
    path = '/media/julian/Volume/data/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/generated/1000/normalized-labels'
    for f in glob.glob(path+'/*/*.h5'):
        #print("clearing flags:", f)
        os.system('h5clear -s ' + f)