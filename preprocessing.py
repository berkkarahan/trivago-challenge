import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('data.csv', delimiter=';')
data.set_index('row_num', inplace=True)

def preparedata(df):
    df['set'] = 'train'
    df.set[df.hits=="\\N"] = 'pred'
    df.session_durantion[df['session_durantion']=="\\N"]=0
    df.session_durantion = df.session_durantion.astype('int')
    df.session_durantion[df.session_durantion==0] = df.session_durantion[df.session_durantion!=0].mean()
    return df

def feature_engineering(df):
    df['count_visited'] = df['path_id_set'].str.split(';').str.len()
    df.count_visited.fillna(0, inplace=True)

    sc = StandardScaler()
    df['sd_scaled'] = sc.fit_transform(df['session_durantion'].values.reshape(-1,1))
    km = KMeans(n_clusters=2, init='k_means++')
    df['session_cluster'] = km.fit_transform(df.sd_scaled.values.reshape(-1,1))
    return df

def encode_features(df):
    dummy = [
        'locale',
        'day_of_week',
        'hour_of_day',
        'agent_id',
        'entry_page',
        'traffic_type',
        'session_cluster'
        ]
    dummies = pd.get_dummies(df[dummy])
    df.drop(dummy, axis=1, inplace=True)
    final_df = pd.merge(df, dummies, left_index=True, right_index=True)
    final_df.drop(['path_id_set'], axis=1, inplace=True)
    final_df['row_num'] = final_df.index
    del dummies
    return final_df

if __name__ == '__main__':
    data = preparedata(data)
    data = feature_engineering(data)
    data = encode_features(data)
    data.to_csv('prep_data.csv', index=False)
