import re

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

    def sort_index(self):
        self.dataframe.sort_index(inplace=True)

    def to_latex(self, output_folder, filename, caption="", label="", description="", long_tables=False, only_tabular_environment=False):
        p = os.path.join(output_folder, filename)
        with pd.option_context("max_colwidth", 1000):
            latex_string = self.dataframe.to_latex(label=label, caption=caption, na_rep='-', float_format="%.3f", bold_rows=True, longtable=long_tables)
        typ = 'longtable' if long_tables else 'tabular'
        latex_string = latex_string.replace(f"\\begin{{{typ}}}", f"\\begin{{adjustbox}}{{width=\\textwidth}}\n\\begin{{{typ}}}")
        latex_string = latex_string.replace(f"\\end{{{typ}}}", f"\\end{{{typ}}}\n\\end{{adjustbox}}")
        latex_lines = latex_string.split('\n')
        if description != "":
            latex_lines = latex_lines[:-1] + [f"\\caption{{{description}}}"] + latex_lines[-1]
        if only_tabular_environment:
            start = 0
            end = len(latex_lines)
            for i, v in enumerate(latex_lines):
                if 'begin{adjustbox}' in v:
                    start = i
                if 'end{adjustbox}' in v:
                    end = i
            latex_lines = latex_lines[start:end+1]
        latex_string = '\n'.join(latex_lines)
        with open(p, 'w') as f:
            f.write(latex_string)
