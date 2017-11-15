import pandas as pd
import numpy as np


def compute_data_table(for_predict=False,workdir='/home/users/adoyle/tadpole/data/'):

    full_df = pd.read_csv(workdir + 'KNN_50_cleaned.csv')
    ref_df = pd.read_csv(workdir + 'KNN_50_cleaned_ref.csv')
    d1d2 = pd.read_csv(workdir + 'TADPOLE_D1_D2.csv')

    full_df.drop(full_df.columns[0], axis=1, inplace=True)
    
    missing_ids = [x for x in ref_df.RID.unique() if any([not pd.notnull(y) for y in ref_df[ref_df.RID==x]['y_DX']])]
    for i,sub in enumerate(missing_ids):
        jnk = ref_df[ref_df.RID==sub]['y_DX'].fillna(method='ffill')
        ref_df.loc[ref_df.RID==sub,'y_DX'] = jnk.values

    full_df = full_df.loc[ref_df.dropna().index].reset_index(drop=True)
    d1d2 = d1d2.loc[ref_df.dropna().index].reset_index(drop=True)
    ref_df = ref_df.loc[ref_df.dropna().index].reset_index(drop=True)

    code = {1: 0, 2:1, 3:2}
    ref_df['y_DX'] = [code[x] for x in ref_df['y_DX']]

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

    if for_predict:
        sel_index = data[ref_df.D2 == 1].index
    else:
        sel_index = data.index

    x_ = []
    y_ = []
    rids = []
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
            rids.append(rid)
            
    return x_, y_, rids

