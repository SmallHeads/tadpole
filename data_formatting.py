import pandas as pd


def compute_data_table(sel_index):
    data_dir = 'C:/Users/adoyle/PycharmProjects/small-heads/data/'

    D1D2_filename = data_dir + 'TADPOLE_D1_D2.csv'
    D3_filename = data_dir + 'TADPOLE_D3.csv'

    encoded_filename = data_dir + 'encoded.csv'

    d1d2 = pd.read_csv(D1D2_filename)
    d3 = pd.read_csv(D3_filename)
    enco = pd.read_csv(encoded_filename, index_col=0)

    data = enco[d3.columns]
    data['Month_bl'] = enco['Month_bl']

    target = data[['DX', 'ADAS13', 'Ventricles', 'ICV']]
    target = target.dropna(axis=0)
    data = data.loc[target.index]
    data = data.dropna(axis=1)
    target = target.loc[data.index]
    target['vent_vol'] = target['Ventricles'].values / target['ICV'].values
    data['vent_vol'] = target['vent_vol']
    target['Month_bl'] = data['Month_bl']
    target = target[['Month_bl', 'DX', 'ADAS13', 'vent_vol']]

    target['DX'].unique()

    target['DX'][target['DX'] == 1] = 0  # NL
    target['DX'][target['DX'] == 3] = 1  # MCI
    target['DX'][target['DX'] == 2] = 2  # AD
    target['DX'][target['DX'] == 5] = 2
    target['DX'][target['DX'] == 6] = 0
    target['DX'][target['DX'] == 7] = 1

    '''
    data['VISCODE'][data['VISCODE']=='bl'] = 0
    data['VISCODE'][data['VISCODE']=='m06'] = 0.6
    data['VISCODE'][data['VISCODE']=='m12'] = 1
    data['VISCODE'][data['VISCODE']=='m18'] = 1.6
    data['VISCODE'][data['VISCODE']=='m24'] = 2
    data['VISCODE'][data['VISCODE']=='m36'] = 3
    data['VISCODE'][data['VISCODE']=='m48'] = 4
    data['VISCODE'][data['VISCODE']=='m60'] = 5
    data['VISCODE'][data['VISCODE']=='m72'] = 6
    data['VISCODE'][data['VISCODE']=='m84'] = 7
    data['VISCODE'][data['VISCODE']=='m96'] = 8
    data['VISCODE'][data['VISCODE']=='m108'] = 9
    data['VISCODE'][data['VISCODE']=='m120'] = 10
    '''
    # 'bl', 'm06', 'm12', 'm24', 'm18', 'm36', 'm48', 'm72', 'm60', 'm84', 'm96', 'm108', 'm120'

    # list_index = target.dropna(axis=0).index

    # target = target.drop(['Ventricles','ICV'],axis=1)
    # data = data.loc[list_index]

    # print(data.columns)

    x_ = []
    y_ = []
    selection = data.iloc[sel_index][['RID', 'Month_bl']]
    for i in range(selection.shape[0]):
        rid, vis_month = selection.iloc[i].values
        # print(rid, vis_month)
        mask = (data[data['RID'] == rid]['Month_bl'] > vis_month).values

        valid_data = data[data['RID'] == rid][mask]
        target_ = target[data['RID'] == rid][mask]

        # print(mask)
        if valid_data.shape[0] > 0:
            # add to x
            x_.append(data.iloc[sel_index[i]].drop(['VISCODE', 'EXAMDATE', 'COLPROT', 'Month_bl']).values)
            # add to y
            target_ = target_.values
            target_[:, 0] = target_[:, 0] - vis_month
            # print(target_)

            y_.append(target_)

    return x_, y_