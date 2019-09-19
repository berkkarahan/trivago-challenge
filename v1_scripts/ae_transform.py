import pandas as pd
import numpy as np
from helpers.keras_autoencoder import AutoEncoderReducer
import gc

from sklearn.model_selection import KFold

data = pd.read_csv('prep_data.csv')
df.set_index(df.row_num, inplace=True)

s = data['set']
data.drop(['set','sd_scaled'], axis=1)

x = data.drop(['hits', 'row_num'], axis=1)
y = data['hits']

xvn = (x.values - np.min(x.values)) / (np.max(x.values) - np.min(x.values))
ind = y.index.values.reshape(-1,1)
del x, data;gc.collect()


def train_encoder_cv(xvn, ind, cv=3):
    kfold = KFold(n_splits=cv)
    firstloop = True
    for tr_i, ho_i in kfold.split(xvn):
        aar = AutoEncoderReducer(xvn[tr_i], xvn[tr_i])
        aar.build()
        aar.summary()
        aar.fit(nb_epoch=8, batch_size=256)
        if firstloop:
            outs = np.divide(aar.transform(xvn), cv)
            outs = np.concatenate((outs, ind), axis=1)
            firstloop = False
        else:
            outs[:,:-1] = outs[:,:-1] + np.divide(aar.transform(xvn),3)
    del xvn, firstloop, ho_i, tr_i;gc.collect()
    return outs

def rebuild_and_save(array, y, s):
    df = pd.merge(pd.DataFrame(outs[:,:-1],index=outs[:,-1].astype('int')),y.to_frame(),left_index=True,right_index=True)
    df = pd.merge(df,s.to_frame(),left_index=True,right_index=True)
    df['row_num'] = df.index
    df.to_csv('transformed_data.csv', index=False)

if __name__ == '__main__':
    arr = train_encoder_cv(xvn, ind)
    rebuild_and_save(arr, y, s)
