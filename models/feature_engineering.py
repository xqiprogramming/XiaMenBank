import numpy as np
import pandas as pd
from os import path

BASE_PATH = path.join(path.dirname(__file__), path.pardir)  ##'D:\\sto\\Project\\XiamenBank'


def load_data(root, start_file_name, fix: list):
    """

    :param root:文件所在的文件夹路径
    :param start_file_name:文件前缀
    :param fix:月份列表
    :return:DateFrame
    """
    base_dir = root + start_file_name
    results = []
    for i in fix:
        try:
            file = "%s%s.csv" % (base_dir, i)
            data = pd.read_csv(file)
            data['month'] = i if i > 6 else i + 12
            results.append(data)
        except Exception as e:
            print(e)
    return pd.concat(results)


# 资产特征
def zichan_feature(df_aum) -> pd.DataFrame:
    """

    :param df_aum: 资产表
    :return:
    """
    df_aum['资产'] = df_aum['X1'] + df_aum['X2'] + df_aum['X3'] + df_aum['X4'] + df_aum['X5'] + df_aum['X6'] + df_aum[
        'X8']
    df_aum['流动资产'] = df_aum['X3'] + df_aum['X4'] + df_aum['X5']
    max_month = df_aum['month'].max()
    # 季度最后一个月 季度内
    features = df_aum[df_aum['month'] == max_month][['cust_no', '资产', 'X7', '流动资产']].copy()
    features['负债率'] = np.divide(features['X7'], features['资产'] + features['X7'])
    # 季度内最后两个月的差异
    df_aum_group = df_aum[df_aum['month'] >= max_month - 1].groupby('cust_no', as_index=False)
    temp = df_aum_group[['资产', 'X7', '流动资产']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'X7': 'X7_last_two_diff',
                 '流动资产': '流动资产_last_two_diff',
                 '资产': '资产_last_two_diff'})
    features = pd.merge(features, temp, how='left')

    # 季度内初始和最后两个月的差异
    df_aum_group = df_aum[(df_aum['month'] == max_month - 2) | (df_aum['month'] == max_month)].groupby('cust_no',
                                                                                                       as_index=False)
    temp = df_aum_group[['资产', 'X7', '流动资产']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'X7': 'X7_start_end_diff',
                 '流动资产': '流动资产_start_end_diff'
            , '资产': '资产_start_end_diff'})
    features = pd.merge(features, temp, how='left')

    # 季度内统计量
    df_aum_group = df_aum[(df_aum['month'] >= max_month - 2)].groupby('cust_no', as_index=False)
    for i in ['max', 'min', 'mean', 'std']:
        temp = df_aum_group[['资产', 'X7', '流动资产']].agg(i).rename(
            columns={'X7': 'X7_' + i, '资产': '资产_' + i, '流动资产': '流动资产_' + i})
        features = pd.merge(features, temp, how='left')

    # 季度间差异：上季度末于本季度末的差异
    df_aum_group = df_aum[(df_aum['month'] == max_month - 3) | (df_aum['month'] == max_month)].groupby('cust_no',
                                                                                                       as_index=False)
    temp = df_aum_group[['资产', 'X7', '流动资产']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'X7': 'X7_quater_diff',
                 '流动资产': '流动资产_quater_diff',
                 '资产': '资产_quater_diff'})
    features = pd.merge(features, temp, how='left')

    # 月度之间差异的统计值，取一个季度
    #     print('月度之间差异的统计值，取一个季度')
    df_jidu = df_aum[df_aum['month'] >= max_month - 2][['cust_no', '资产', 'X7', '流动资产']].copy()
    df_aum_group = df_jidu.groupby('cust_no')
    for col in ['资产', 'X7', '流动资产']:
        temp = df_aum_group[col].diff().to_frame(col + '_consist_diff')
        df_jidu = pd.concat([df_jidu, temp], axis=1, ignore_index=False)
    df_aum_group = df_jidu[df_jidu[col + '_consist_diff'].notnull()].groupby('cust_no', as_index=False)
    for col in ['资产', 'X7', '流动资产']:
        for i in ['mean', 'max', 'min', 'std']:
            s = df_aum_group[col + '_consist_diff'].agg(i).rename(
                columns={col + '_consist_diff': col + '_consist_diff' + '_' + i})
            features = pd.merge(features, s, how='left')
    return features


# 存款特征
def cunkuan_feature(df) -> pd.DataFrame:
    """

    :param df: 存款表
    :return:
    """
    max_month = df['month'].max()
    temp = df[df['month'] == max_month]
    # 季度最后一个月
    features = df[df['month'] == max_month][['cust_no', 'C1', 'C2']].copy()
    # 季度内最后两个月的差异
    df_group = df[df['month'] >= max_month - 1].groupby('cust_no', as_index=False)
    temp = df_group[['C1', 'C2']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'C1': 'C1_last_two_diff', 'C2': 'C2_last_two_diff'})
    features = pd.merge(features, temp, how='left')

    # 季度内初始和最后一个月的差异
    df_group = df[(df['month'] == max_month - 2) | (df['month'] == max_month)].groupby('cust_no', as_index=False)
    temp = df_group[['C1', 'C2']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'C1': 'C1_start_end_diff', 'C2': 'C2_start_end_diff'})
    features = pd.merge(features, temp, how='left')

    # 季度间差异：上季度末于本季度末的差异
    df_group = df[(df['month'] == max_month - 3) | (df['month'] == max_month)].groupby('cust_no', as_index=False)
    temp = df_group[['C1', 'C2']].apply(lambda x: x.diff().iloc[-1, :]).rename(
        columns={'C1': 'C1_quater_diff', 'C2': 'C2_quater_diff'})
    features = pd.merge(features, temp, how='left')

    # 季度内统计值

    # 半年内统计值
    return features


# 月度账户行为特征
def behaviour_feature(df, end_day: str):
    """

    :param df: 月度账户行为表
    :param end_day: 季度末日期+1，即下一季度开始日期
    :return:
    """
    df = df.copy()
    df['流水'] = df['B3'] + df['B5']
    df['净转入'] = df['B3'] - df['B5']
    max_month = df['month'].max()
    end_day = pd.to_datetime(end_day)
    # 最近一次交易时间距离本季度末时间差
    features = df[df['month'] == max_month][['cust_no', 'B6']]
    features['B6'] = features['B6'].apply(lambda x: (end_day - pd.to_datetime(x)).days)
    # 季度间变化
    pre_q = df[df['month'] == max_month - 3][['cust_no', 'B7']].copy().rename(columns={'B7': 'B7_pre_q'})
    temp = pd.merge(df[df['month'] == max_month][['cust_no', 'B7']], pre_q, how='left')
    temp['B7_q_diff'] = temp['B7'] - temp['B7_pre_q']
    features = pd.merge(features, temp[['cust_no', 'B7_q_diff']], how='left')
    # 季度末最后一个月的值
    features = pd.merge(features,
                        df[df['month'] == max_month][['cust_no', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', '流水', '净转入']])

    # 季度内流水/登录/转入/转出统计量
    df_group = df[df['month'] >= max_month - 2].groupby('cust_no', as_index=False)
    for col in ['流水', 'B1', 'B2', 'B4', '净转入']:
        for i in ['sum', 'mean']:
            temp = df_group[col].agg(i).rename(columns={col: col + '_' + i})
            features = pd.merge(features, temp, how='left')

    # 半年内流水/登录/转入/转出统计量
    df_group = df.groupby('cust_no', as_index=False)
    for col in ['流水', 'B1', 'B2', 'B4', '净转入']:
        for i in ['sum', 'mean', 'max', 'min', 'std']:
            temp = df_group[col].agg(i).rename(columns={col: col + '_6m_' + i})
            features = pd.merge(features, temp, how='left')
    # 最近一个季度的CTR
    features['CTR'] = features['B7'] / (features['B1_sum'] + 0.0001)
    return features


# 重大事件特征
def big_event_feature(df, end_day: str):
    """

    :param df: 重大事件表
    :param end_day: 季度末日期+1，即下一季度开始日期
    :return:
    """
    df = df.copy()
    df = df.drop(columns=['E3', 'E7', 'E8', 'E9', 'E11', 'E12', 'E13'])
    for col in ['E1', 'E2', 'E4', 'E5', 'E6', 'E10',
                'E14', 'E16', 'E18']:
        df[col] = df[col].astype('datetime64[ns]')
        idx = df[(df[col] < pd.to_datetime('2015-3-28')) | (df[col] > pd.to_datetime(end_day))].index
        df.loc[idx, col] = np.nan
    end_day = pd.to_datetime(end_day)
    df['end_day'] = end_day
    features = df[['cust_no', 'E15', 'E17']].copy()
    # 所有重大事件日期与下一季度第一天的时间差
    for col in ['E1', 'E2', 'E4', 'E5', 'E6', 'E10', 'E14', 'E16', 'E18']:
        features['%s_diff' % col] = df['end_day'] - df[col]
        features['%s_diff' % col] = features['%s_diff' % col].apply(lambda x: x.days)
    return features


# 客户属性特征
def info_feature(df):
    """

    :param df: 客户属性表
    :return:
    """
    df = df.copy()
    I1_map = {'男性': 1, '女性': 0}
    I3_map = {'普通客户': 0, '黄金': 1, '白金': 2, '钻石': 3}
    for col in ['I4', 'I5', 'I7', 'I8', 'I12', 'I15', 'I17', 'I19', 'I9', 'I10', 'I11', 'I13', 'I14', 'month']:
        try:
            df.drop(columns=col, inplace=True)
        except Exception:
            pass
    # 年龄分箱
    df['I2'] = pd.cut(df['I2'], [0, 17, 29, 39, 49, 59, 100], labels=[1, 2, 3, 4, 5, 6])
    df['I1'] = df['I1'].map(I1_map)
    df['I3'] = df['I3'].map(I3_map)
    return df


# 特征工程
def feature_engneering(df_aum, df_cunkuan, df_behaviour, df_big_event, cust_info, label_pre_q, end_day):
    """

    :param df_aum: 资产
    :param df_cunkuan:存款
    :param df_behaviour: 行为
    :param df_big_event: 重大事件
    :param cust_info: 属性
    :param label_pre_q: 上季度Label
    :param end_day: 季度末日期+1，即下一季度开始日期
    :return:
    """
    label_pre_q = label_pre_q.copy()
    label_pre_q.rename(columns={'label': 'label_pre_q'}, inplace=True)
    zichan_feat = zichan_feature(df_aum)
    cunkuan_feat = cunkuan_feature(df_cunkuan)
    behav_feat = behaviour_feature(df_behaviour, end_day)
    big_ev_feat = big_event_feature(df_big_event, end_day)
    cust_feat = info_feature(cust_info)
    features = pd.merge(big_ev_feat, behav_feat, how='left')
    features = pd.merge(features, cunkuan_feat, how='left')
    features = pd.merge(features, zichan_feat, how='left')
    features = pd.merge(features, label_pre_q, how='left')
    features = pd.merge(features, cust_feat, how='left')
    features['label_pre_q'] = features['label_pre_q'].fillna(2)
    # 处理缺失值
    try:
        for col in ['E4_diff', 'E5_diff', 'E16_diff', 'E18_diff']:
            features[col] = features[col].fillna(99999)
    except Exception:
        pass
    return features


def load_train_data():
    try:
        print('正在读取文件')
        features = pd.read_excel(path.join(BASE_PATH, 'data', 'clean', 'features.xlsx'))
        print('读取完成')
    except FileNotFoundError:
        print('文件不存在，正在新建特征文件！')
        data_root = path.join(BASE_PATH, 'data')
        data_x_root = path.join(BASE_PATH, 'x_train')
        data_y_root = path.join(BASE_PATH, 'y_train_3')
        fix = [7, 8, 9, 10, 11, 12]
        print('正在读取原始文件数据')
        cust_q3 = pd.read_csv(path.join(BASE_PATH, 'x_train', "cust_avli_Q3.csv"))
        cust_q4 = pd.read_csv(path.join(BASE_PATH, 'x_train', "cust_avli_Q4.csv"))
        lable = pd.read_csv(path.join(data_y_root, 'y_Q4_3.csv'))
        lable_q3 = pd.read_csv(path.join(data_y_root, 'y_Q3_3.csv'))
        aum = load_data(data_x_root + 'aum_train/', 'aum_m', fix)
        aum = pd.merge(aum, cust_q4, how='inner')
        behavior = load_data(data_x_root + 'behavior_train/', 'behavior_m', fix)
        behavior = pd.merge(behavior, cust_q4, how='inner')
        cunkuan = load_data(data_x_root + 'cunkuan_train/', 'cunkuan_m', fix)
        cunkuan = pd.merge(cunkuan, cust_q4, how='inner')
        fix = [4]
        big_event = pd.read_csv(path.join(data_x_root, 'big_event_train', 'big_event_Q4.csv'))
        big_event = pd.merge(big_event, cust_q4, how='inner')
        cust_info = load_data(data_x_root, 'cust_info_q', fix)
        cust_info = pd.merge(cust_info, cust_q4, how='inner')
        cust_info = pd.merge(cust_info, lable, how='left')
        print('正在提取特征')
        features = feature_engneering(aum, cunkuan, behavior, big_event, cust_info, lable_q3, '2020-01-01')
        print('正在存储特征文件')
        features.to_excel(path.join(data_root, 'clean', 'features.xlsx'), index=False)
        print('存储完成')


def load_test_data():
    data_root = path.join(BASE_PATH, 'data')
    data_x_root = path.join(BASE_PATH, 'x_train')
    data_y_root = path.join(BASE_PATH, 'y_train_3')

    try:
        print('正在读取test特征文件')
        features = pd.read_excel(path.join(data_root, 'clean', 'features_test.xlsx'))
        print('读取完成')
    except FileNotFoundError:
        print('文件不存在，正在新建特征文件！')
        fix = [10, 11, 12, 1, 2, 3]
        print('正在读取原始文件数据')
        cust_q3 = pd.read_csv(path.join(BASE_PATH, 'x_train', "cust_avli_Q4.csv"))
        cust_q4 = pd.read_csv(path.join(BASE_PATH, 'x_train', "cust_avli_Q1.csv"))
        lable_q3 = pd.read_csv(path.join(data_y_root, 'y_Q4_3.csv'))
        aum = load_data(data_x_root + 'aum_train/', 'aum_m', fix)
        aum = pd.merge(aum, cust_q4, how='inner')
        behavior = load_data(data_x_root + 'behavior_train/', 'behavior_m', fix)
        behavior = pd.merge(behavior, cust_q4, how='inner')
        cunkuan = load_data(data_x_root + 'cunkuan_train/', 'cunkuan_m', fix)
        cunkuan = pd.merge(cunkuan, cust_q4, how='inner')
        fix = [1]
        big_event = pd.read_csv(path.join(data_x_root, 'big_event_train', 'big_event_Q1.csv'))
        big_event = pd.merge(big_event, cust_q4, how='inner')
        cust_info = load_data(data_x_root, 'cust_info_q', fix)
        cust_info = pd.merge(cust_info, cust_q4, how='inner')
        print('正在提取特征')
        features = feature_engneering(aum, cunkuan, behavior, big_event, cust_info, lable_q3, '2020-04-01')
        print('正在存储特征文件')
        features.to_excel(path.join(data_root, 'clean', 'features_test.xlsx'), index=False)
        print('存储完成')


def main(is_train=True):
    if is_train:
        load_train_data()
    else:
        load_test_data()


if __name__ == '__main__':
    main()
    exit(0)
