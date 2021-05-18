import pandas as pd
import os

class DataFrameFactory():
    def __init__(self, data=None, colum_labels=None, dataframe:pd.DataFrame=None, index_name=None):
        self.columns = colum_labels
        self.dataframe = None
        if data is None:
            if dataframe is not None:
                self.dataframe = pd.DataFrame(dataframe, columns=colum_labels)
        else:
            self.dataframe = pd.DataFrame(data, columns=colum_labels)

    def append(self, data):
        if self.dataframe is None:
            self.dataframe = pd.DataFrame(data, self.columns)
        else:
            dftemp = pd.DataFrame(data, columns=self.columns)
            self.dataframe = self.dataframe.append(dftemp)

    def get_dataframe(self):
        return self.dataframe

    def __str__(self):
        return self.dataframe.__str__()

    def to_csv(self, output_folder, filename):
        p = os.path.join(output_folder, filename)
        self.dataframe.to_csv(p)

    def to_latex(self, output_folder, filename, caption="", label=""):
        p = os.path.join(output_folder, filename)
        latex_string = self.dataframe.to_latex(label=label, caption=caption)
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write(latex_string)
