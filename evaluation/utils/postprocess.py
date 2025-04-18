import pandas as pd
import numpy as np

def post_average(metric_file):
    df = pd.read_csv(metric_file, index_col=False)
    # Remove any existing 'Average' rows
    df = df[df['Dataset'] != 'Average']
    df = df.drop_duplicates(['Dataset'])
    mean_row = df.select_dtypes(include=[np.number]).mean().round(3)
    new_row = pd.DataFrame([['Average']], columns=['Dataset']).join(pd.DataFrame([mean_row.values], columns=mean_row.index))
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(metric_file, index=False)
