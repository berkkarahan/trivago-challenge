import pandas as pd
import numpy as np
import xgboos as xgb
import swifter
import gc

from scipy.stats import kurtosis, skew
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.base import clone

data = pd.read_csv('transformed_data.csv')
data.index = data['row_num']
data.drop(['row_num'], axis=1, inplace=True)

train = data[data['set']=='train'].copy()
pred = data[data['set']=='pred'].copy()

train.drop(['set'], axis=1, inplace=True)
pred.drop(['set'], axis=1, inplace=True)
pred.drop(['hits'], axis=1, inplace=True)

train['hits'] = train.hits.astype('int')


def build_ftrs(df):
    print("Building mean.")
    df['r_wise_mean'] = df.swifter.apply(lambda row:np.mean(row), axis=1)
    print("Building median.")
    df['r_wise_median'] = df.swifter.apply(lambda row:np.median(row), axis=1)
    print("Building standard deviation.")
    df['r_wise_std'] = df.swifter.apply(lambda row:np.std(row), axis=1)
    print("Building skewness.")
    df['r_wise_skew'] = df.swifter.apply(lambda row:kurtosis(row), axis=1)
    print("Building kurtosis.")
    df['r_wise_kurt'] = df.swifter.apply(lambda row:skew(row), axis=1)
    return df


def cv_get_outliers(x, y, nfolds=3):
    def _check_inlier(val):
        if val > 0:
            return 1
        elif val < 0:
            return 0
        elif val == 0:
            return 0

    clf = IsolationForest(n_jobs=-1)
    kf = KFold(n_splits=nfolds)
    firstrun = True
    i = 1
    for tr_i, _ in kf.split(x,y):
        print("{}-th run.".format(i))
        clf_ = clone(clf)
        if firstrun:
            firstrun = False
            clf_.fit(x.iloc[tr_i,:], y.to_frame().iloc[tr_i,:])
            inlierdf = pd.DataFrame({'Index':x.index,
                                  'IsInlier':clf_.predict(x)})
        else:
            clf_.fit(x.iloc[tr_i,:], y.to_frame().iloc[tr_i,:])
            inlierdf['IsInlier'] = (inlierdf['IsInlier'] + clf_.predict(x))/2
        i = i + 1

    inlierdf['final_sts'] = inlierdf.swifter.apply(lambda row: _check_inlier(row['IsInlier']), axis=1)
    inlierdf.set_index('Index', inplace=True)
    inlierdf.drop(['IsInlier'], axis=1, inplace=True)
    x = pd.merge(x, inlierdf, right_index=True, left_index=True)
    y = pd.merge(y.to_frame(), inlierdf, right_index=True, left_index=True)
    x = x[x['final_sts']==1]
    x.drop(['final_sts'],axis=1,inplace=True)
    y = y[y['final_sts']==1]
    y.drop(['final_sts'],axis=1,inplace=True)

    return (x, y)


def cv_get_importances(clf, x, y, nfolds=3):
    kf = KFold(n_splits=nfolds)
    firstrun = True
    feat_labels = list(x.columns.values)
    for tr_i, _ in kf.split(x, y):
        clf_ = clone(clf)
        if firstrun:
            clf_.fit(x.iloc[tr_i,:], y.iloc[tr_i,:])
            impdf = pd.DataFrame({'FeatureLabels':feat_labels,
                                  'Importances':clf_.feature_importances_})
            impdf['Importances'] = impdf['Importances']/nfolds
        else:
            impdf['Importances'] = impdf['Importances'] + np.divide(clf_.feature_importances_,3)
    return impdf


def pareto(imp, perc=0.8):
    imp['percentage'] = imp.apply(lambda row: row['Importances']/np.sum(imp['Importances']), axis=1)
    imp = imp[imp['percentage']!=0]
    imp.sort_values(by=['percentage'], ascending=False, inplace=True)
    imp['cml_p'] = imp['percentage'].cumsum()
    imp['valid'] = imp.apply(lambda row: True if row.cml_p <= np.float(0.8) else False, axis=1)
    return list(imp[imp.valid==True].FeatureLabels.values)


def rebuild_and_save(x_rft, y, pred_rft):
    x_rft['row_num'] = x.index
    pred_rft['row_num'] = pred.index
    y['row_num'] = y.index
    x_rft.to_csv('rft_x.csv', index=False)
    pred_rft.to_csv('rft_pred.csv', index=False)
    y.to_csv('rft_y.csv', index=False)


if __name__ == '__main__':
    x = train.drop(['hits'], axis=1)
    y = train['hits']

    x.fillna(x.mean(), inplace=True)
    x = build_ftrs(x)
    pred = build_ftrs(pred)

    yv = np.log1p(y)
    x, y = cv_get_outliers(x, yv)

    xg = xgb.XGBRegressor(n_jobs=-1)
    imp = cv_get_importances(xg, x, y)
    valftr = pareto(imp)

    x_rft = x[valftr].copy()
    pred_rft = pred[valftr].copy()

    rebuild_and_save(x_rft, y, pred_rft)
