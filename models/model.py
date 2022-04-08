from copy import deepcopy
from functools import reduce
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import lightgbm as lgb
import catboost as catb
import pickle
from os import path
BASE_PATH = path.join(path.dirname(__file__), path.pardir)



def search_model(model, params, X, y, scoring, cv, weight=None):
    print('==========%s==========' % 'GridSearch')
    select_cv = ms.GridSearchCV(estimator=model, param_grid=params, scoring=scoring, cv=cv, verbose=1)
    select_cv.fit(X=X, y=y.values.ravel())
    print('==========%s==========' % 'Best Params')
    print(select_cv.best_params_)
    return select_cv.best_params_


def kappa_score(x, y, z):
    pred = x.predict(y)
    return metrics.cohen_kappa_score(z, pred)


def eval_func(y_true, y_pred):
    y_pred = np.reshape(y_pred, (3, -1)).T.argmax(axis=1)
    sc = metrics.cohen_kappa_score(y_true, y_pred)
    return 'eval', sc, True


def model_stack(model_list: list, params_list: list, model_stacking, params_stacking, train_set, valid_set, scoring,
                is_search: bool = True,
                random_state=0):
    """

    :param model_list: 模型列表
    :param params_list: 模型列表对应的超参数列表，当is_search=True时，需要设置为参数搜索模式，为False时，为模型参数设置模式
    :param model_stacking: 最后融合的模型
    :param params_stacking: 融合模型的参数搜索模型
    :param train_set: 训练数据，格式为（X,y）
    :param valid_set: 训练时的验证集，格式为(X,y)
    :param scoring: GridSearch的评估函数
    :param is_search: 设置是否需要最优参数搜索
    :param random_state: 随机状态
    :return:
    """
    # 先寻找超参数
    n = len(model_list)
    cv = ms.StratifiedKFold(5)
    X, y = train_set
    results_models = []
    level_2_data = []
    for i in range(n):
        print('============训练第%s类模型===========' % (i + 1))
        params = params_list[i]
        model = deepcopy(model_list[i])
        if is_search:
            best_params = search_model(model, params, X, y, scoring, cv)
        else:
            best_params = params
        # 第一层预测
        model = model.set_params(**best_params)
        models, oof_data = get_oof(model, X, y, cv, valid_set)
        level_2_data.append(oof_data)
        results_models.append(models)
    level_2_data = reduce(lambda x, y: np.concatenate([x, y], axis=1), level_2_data)
    # 第二层训练
    print('============第二层训练=============')
    best_params = search_model(model_stacking, params_stacking, level_2_data, y, scoring, cv)
    model_stacking = model_stacking.set_params(**best_params)
    model_stacking.fit(level_2_data, y)
    return {'results_models': results_models, 'stacking_model': model_stacking, 'level_2_X': level_2_data,
            'level_2_y': y}


def get_oof(model, X, y, cv, valid_set):
    n_class = len(y.unique())
    oof_train = np.zeros((X.shape[0], n_class))
    model_list = []
    for j, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        cv_train_x = X.iloc[train_idx]
        cv_train_y = y.iloc[train_idx]
        cv_valid_x = X.iloc[valid_idx]
        cv_valid_y = y.iloc[valid_idx]
        if isinstance(model, lgb.LGBMClassifier):
            callbacks = [lgb.callback.early_stopping(100, verbose=True)]
            model.fit(cv_train_x, cv_train_y, eval_set=[valid_set],
                      eval_metric=eval_func, callbacks=callbacks, verbose=False)
        elif isinstance(model, catb.CatBoostClassifier):
            model.fit(cv_train_x, cv_train_y, eval_set=catb.Pool(*valid_set))
        else:
            model.fit(cv_train_x, cv_train_y)
        oof_train[valid_idx] = model.predict_proba(cv_valid_x)
        model_list.append(model)
    return model_list, oof_train


def predict(model_list, stacking_model, X, y=None):
    predicts = []
    for i in range(len(model_list)):
        temp = np.zeros((X.shape[0], 3))
        for model in model_list[i]:
            temp += model.predict_proba(X)
        temp /= 3
        predicts.append(temp)
    predicts = reduce(lambda x, y: np.concatenate([x, y], axis=1), predicts)
    pred = stacking_model.predict(X=predicts)
    return pred


def main(is_train=True):
    fix = '' if is_train else '_test'
    try:
        print('正在读取特征文件！')
        path.join(BASE_PATH, 'data', 'clean', 'features' + fix + '.xlsx')
        data = pd.read_excel(path.join(BASE_PATH, 'data', 'clean', 'features' + fix + '.xlsx'))
    except FileNotFoundError:
        raise ('没有找到数据文件！')
    rdm = 2022
    data['label_pre_q'] = data['label_pre_q'] + 1
    feat_cols = data.columns
    y = data.columns[-1]
    for col in ['B2_6m_min', 'B4_6m_sum', 'C2_last_two_diff', 'X7_start_end_diff',
                'X7_std', 'X7_consist_diff_max', 'X7_consist_diff_min', 'I6',
                'I16', 'I18', 'I20', 'cust_no', 'label']:
        try:
            feat_cols = feat_cols.drop(col)
        except Exception as e:
            pass
    train_x, eval_x, train_y, eval_y, test_x, test_y = None, None, None, None, None, None
    if is_train:
        data['label'] = data['label'] + 1
        train_x, test_x, train_y, test_y = ms.train_test_split(data[feat_cols], data[y], train_size=0.7,
                                                               stratify=data[y], random_state=rdm)
        train_x, eval_x, train_y, eval_y = ms.train_test_split(train_x, train_y, train_size=0.7, stratify=train_y,
                                                               random_state=rdm)

    try:
        final_model = pickle.load(open(path.join(BASE_PATH, 'models', 'final_model.pickle'), 'rb'))
    except FileNotFoundError:
        if not is_train:
            raise ('预测模式，但未找到模型文件!')
        print('没有找到模型文件,正在构建模型')
        model_list = [lgb.LGBMClassifier(n_estimators=1000, subsample=0.6, colsample_bytree=0.6, random_state=rdm,
                                         objective='multiclass'),
                      catb.CatBoostClassifier(learning_rate=0.05, random_seed=rdm, od_type='Iter', od_wait=100,
                                              objective='MultiClass',
                                              n_estimators=2000, eval_metric='Kappa', verbose=200,
                                              colsample_bylevel=0.6)
                      ]
        parmas_lgb = {'max_depth': 5, 'num_leaves': 31, 'reg_lambda': 1,
                      'reg_alpha': 1}
        parmas_cat = {'depth': 6, 'l2_leaf_reg': 5}
        parmas_list = [parmas_lgb, parmas_cat]
        model_stacking = LogisticRegression(random_state=rdm, max_iter=1000, class_weight='balanced')
        params_stacking = {'C': [1, 2, 4, 0.5]}
        final_model = model_stack(model_list, parmas_list, model_stacking, params_stacking, (train_x, train_y),
                                  (eval_x, eval_y), kappa_score, False, rdm)
        print('正在存储模型')
        pickle.dump(final_model, open(path.join(BASE_PATH, 'models', 'final_model.pickle'), 'wb'))
        print('存储完成')
    if is_train:
        test_pred = predict(final_model['results_models'], final_model['stacking_model'], test_x)
        print('测试Kappa分数：', cohen_kappa_score(test_pred, test_y))
    else:
        test_pred = predict(final_model['results_models'], final_model['stacking_model'], data[feat_cols])
        pred = data[['cust_no']].copy()
        pred['pred'] = test_pred - 1
        pred.to_excel(path.join(BASE_PATH, 'predict', 'pred.xlsx'), index=False)
        print('存储完成')


if __name__ == '__main__':
    main(False)
    exit(0)
    # final_model = pickle.load(open('./final_model.pickle', 'rb'))
