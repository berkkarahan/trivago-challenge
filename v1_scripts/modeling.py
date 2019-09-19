import pandas as pd
import numpy as np
import threading
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import gc

import xgboost as xgb

x = pd.read_csv('rft_x.csv')
x.set_index('row_num', inplace=True)
y = pd.read_csv('rft_y.csv')
trg = y.hits.values.reshape(-1,1)
pred = pd.read_csv('rft_pred.csv')
pred.set_index('row_num', inplace=True)


def fit_model_cv(mdl, x, y, cv=5):

    def _fitmodelxgb(mdl, x, y, tr_i, ho_i):
        eval_set = [(x.iloc[ho_i,:],y[ho_i])]
        p = mdl.get_params()
        p['n_jobs'] = -1
        p['n_estimators'] = 250
        mdl.set_params(**p)
        del p;gc.collect()
        return mdl.fit(x.iloc[tr_i,:],y[tr_i], early_stopping_rounds=15, eval_set=eval_set, eval_metric='rmse', verbose=True)

    def _fitmodel(mdl,x,y, tr_i, ho_i):
        return mdl.fit(x.iloc[tr_i,:],y[tr_i])

    kfold = KFold(n_splits=cv)
    threadlist = []
    modlist = []
    train_ind = []
    holdout_ind= []
    predictions = []
    score = []

    for tr_i, ho_i in kfold.split(x,y):
        train_ind.append(tr_i)
        holdout_ind.append(ho_i)
        cloned_mdl = clone(mdl)
        modlist.append(cloned_mdl)
        if isinstance(cloned_mdl, xgb.XGBRegressor):
            task = threading.Thread(target=_fitmodelxgb, args=(cloned_mdl, x, y, tr_i, ho_i, ), daemon=True)
        else:
            task = threading.Thread(target=_fitmodel, args=(cloned_mdl, x, y, tr_i, ho_i, ), daemon=True)
        threadlist.append(task)

    for t in threadlist:
        t.start()

    for t in threadlist:
        t.join()

    for i in range(0,5):
        m = modlist[i]
        predictions.append(m.predict(x.iloc[holdout_ind[i],:]))
        score.append(mean_squared_error(y[holdout_ind[i]], predictions[i]))
    del threadlist, train_ind, holdout_ind, predictions
    gc.collect()
    print("Average mean_sq_error for models are: {}".format(np.mean(score)))
    return modlist


def test_rounding_cv(mlist, x, y, rounding='up'):
    preds = []
    scores = []
    for m in mlist:
        ypred = np.expm1(m.predict(x))
        if rounding=='up':
            ypred=np.ceil(ypred)
            scores.append(mean_squared_error(y,ypred))
        elif rounding=='down':
            ypred=np.floor(ypred)
            scores.append(mean_squared_error(y,ypred))
    print("For strategy {}, the mean_squared_error is: {}".format(rounding, np.mean(scores)))
    return np.mean(scores)


def make_predictions(mlist, x, rounding='up'):
    import swifter
    import datetime
    pred_df=pd.DataFrame()
    for i, m in enumerate(mlist):
        pred_df['pred_'+str(i)] = np.expm1(m.predict(x))
    pred_df.set_index(x.index, inplace=True)
    pred_df['hits'] = pred_df.swifter.apply(lambda row:np.mean(row), axis=1)
    if rounding=='up':
        pred_df['hits'] = np.ceil(pred_df['hits'])
    elif rounding=='down':
        pred_df['hits'] = np.floor(pred_df['hits'])
    pred_df['row_num'] = pred_df.index
    finalpred = pred_df[['row_num','hits']].copy()
    now = datetime.datetime.now()
    filestr = 'submission_' + str(now.day) + '_' + str(now.month) + '_' + str(now.year) + '.csv'
    finalpred.to_csv(filestr, index=False)
    del pred_df;gc.collect()
    return finalpred


if __name__ == '__main__':

    defaultparams = xgb.XGBRegressor().get_params()

    p1 = dict(defaultparams)
    p1['learning_rate'] = 0.2
    p1['max_depth'] = 3
    p1['subsample'] = 0.5
    p1['n_jobs'] = -1
    p1_xgb = xgb.XGBRegressor()
    p1_xgb.set_params(**p1)

    p2 = dict(defaultparams)
    p2['learning_rate'] = 0.01
    p2['max_depth'] = 10
    p2['subsample'] = 0.7
    p2['n_jobs'] = -1
    p2_xgb = xgb.XGBRegressor()
    p2_xgb.set_params(**p2)

    p3 = dict(defaultparams)
    p3['learning_rate'] = 0.05
    p3['max_depth'] = 8
    p3['subsample'] = 0.9
    p3['n_jobs'] = -1
    p3_xgb = xgb.XGBRegressor()
    p3_xgb.set_params(**p3)

    p4 = dict(defaultparams)
    p4['learning_rate'] = 0.1
    p4['max_depth'] = 5
    p4['subsample'] = 0.8
    p4['n_jobs'] = -1
    p4_xgb = xgb.XGBRegressor()
    p4_xgb.set_params(**p4)

    p5 = dict(defaultparams)
    p5['learning_rate'] = 0.4
    p5['max_depth'] = 7
    p5['subsample'] = 0.6
    p5['n_jobs'] = -1
    p5_xgb = xgb.XGBRegressor()
    p5_xgb.set_params(**p5)

    p1_l = fit_model_cv(p1_xgb, x, trg) #Average mean_sq_error for models are: 0.2631375318644249
    p2_l = fit_model_cv(p2_xgb, x, trg) #Average mean_sq_error for models are: 0.711170601153347
    p3_l = fit_model_cv(p3_xgb, x, trg) #Average mean_sq_error for models are: 0.2570784015395205
    p4_l = fit_model_cv(p4_xgb, x, trg) #Average mean_sq_error for models are: 0.2603705463442478
    p5_l = fit_model_cv(p5_xgb, x, trg) #Average mean_sq_error for models are: 0.2630508185598705

    trg_tru = np.expm1(trg)

    fl = test_rounding_cv(p3_l, x, trg_tru, rounding='down') #For strategy down, the mean_squared_error is: 420.28449800726656
    cl = test_rounding_cv(p3_l, x, trg_tru, rounding='up') #For strategy up, the mean_squared_error is: 414.1223534738968

    out = make_predictions(p3_l2, pred, rounding='down')
