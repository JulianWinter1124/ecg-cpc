import glob, os

from util import ecg_data

if __name__ == '__main__':
    path = '/media/julian/Volume/data/ECG/ptb-diagnostic-ecg-database-1.0.0/generated/normalized'
    for f in glob.glob(path+'/*/*.h5'):
        print("clearing flags:", f)
        os.system('h5clear -s ' + f)