import pandas as pd
import numpy as np


def compute_data_table():
    workdir = 'C:/Users/adoyle/PycharmProjects/small-heads/data/'

    full_df = pd.read_csv(workdir + 'inputed50_cleaned.csv')
    ref_df = pd.read_csv(workdir + 'inputed50_ref.csv')
    d1d2 = pd.read_csv(workdir + 'TADPOLE_D1_D2.csv')

    full_df.drop(full_df.columns[0], axis=1, inplace=True)

    full_df['RID'] = ref_df['RID']
    full_df['VISCODE'] = ref_df['VISCODE']
    full_df['COLPROT'] = ref_df['COLPROT']
    full_df['EXAMDATE'] = ref_df['EXAMDATE']
    full_df['Month_bl'] = d1d2['Month_bl']

    data = full_df

    target = ref_df[['y_DX', 'y_ADAS13', 'y_Ventricles_adj_TIV']]
    # target = target.dropna(axis=0)
    # data = data.loc[target.index]
    # data = data.dropna(axis=1)
    # target = target.loc[data.index]
    target['Month_bl'] = data['Month_bl']
    target = target[['Month_bl', 'y_DX', 'y_ADAS13', 'y_Ventricles_adj_TIV']]

    for col in target.columns[1:]:
        data[col] = target[col]

    sel_index = data.index

    x_ = []
    y_ = []
    selection = data.iloc[sel_index][['RID','Month_bl']]
    for i in range(selection.shape[0]):
        rid, vis_month = selection.iloc[i].values
        #print(rid,vis_month)
        mask = (data[data['RID']==rid]['Month_bl']>vis_month).values

        valid_data = data[data['RID']==rid][mask]
        target_ = target[data['RID']==rid][mask]
        
        #print(mask)
        if valid_data.shape[0]>0:
            # add to x 
            x_.append(data.iloc[sel_index[i]].drop(['VISCODE','EXAMDATE','COLPROT','Month_bl']).values)
            # add to y
            target_ = target_.values
            target_[:,0] = target_[:,0]-vis_month
            #print(target_)
            
            y_.append(target_)
            
    return x_, y_

