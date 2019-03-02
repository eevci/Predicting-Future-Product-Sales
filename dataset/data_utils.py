
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce

def read_data(parentPath):
    sales = preprocess3(parentPath)
    train_set = sales
    train_x_set = []
    train_y_set = []
    val_x_set = []
    val_y_set = []
    test_x_set = []
    test_y_set = []

    for row in range(0,30):

        train_y_set.append(train_set[train_set['date_block_num'] == row+1].values[:10000, -1])
        x = pd.DataFrame(data=train_set[train_set['date_block_num'] == row].values[:10000, :],
                             columns=list(train_set.columns))
        x = x.drop(columns=['item_category','item_id','shop_id'])
        train_x_set.append(x.values)
    train_x_set = np.array(train_x_set)
    train_y_set = np.array(train_y_set)
    train_x_set = np.transpose(train_x_set, (1, 0, 2))
    train_y_set = np.transpose(train_y_set, (1, 0))

    for row in range(1,31):

        val_y_set.append(train_set[train_set['date_block_num'] == row+1].values[:10000, -1])
        x = pd.DataFrame(data=train_set[train_set['date_block_num'] == row].values[:10000, :],
                             columns=list(train_set.columns))
        x = x.drop(columns=['item_category','item_id','shop_id'])
        val_x_set.append(x.values)
    val_x_set = np.array(val_x_set)
    val_y_set = np.array(val_y_set)
    val_x_set = np.transpose(val_x_set, (1, 0, 2))
    val_y_set = np.transpose(val_y_set, (1, 0))

    for row in range(2,32):

        test_y_set.append(train_set[train_set['date_block_num']==row+1].values[:10000, -1])
        x = pd.DataFrame(data=train_set[train_set['date_block_num'] == row].values[:10000, :],
                         columns=list(train_set.columns))
        x = x.drop(columns=['item_category','item_id','shop_id'])
        test_x_set.append(x.values)
    test_x_set = np.array(test_x_set)
    test_y_set = np.array(test_y_set)
    test_x_set = np.transpose(test_x_set, (1, 0, 2))
    test_y_set = np.transpose(test_y_set, (1, 0))

    return train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set

def readData2(parentPath, windowSize):
    sales = preprocess3(parentPath)
    test_split = .20
    val_split = .10
    N = sales.shape[0]
    train_size = int(N * (1 - test_split))
    val_size = int(train_size * val_split)
    train_size = train_size - val_size
    test_size = N - train_size
    train_set = sales[sales['date_block_num'] < 13]

    train_x_set = []
    train_y_set = []
    val_x_set = []
    val_y_set = []
    test_x_set = sales[sales['date_block_num'] == 32].values
    test_y_set = sales[sales['date_block_num'] == 33].values[:, -1]
    count = 0
    for row in range(1,12):
        train_y_set.append(train_set[train_set['date_block_num'] == row].values[:10000, -1])
        train_x_set.append(train_set[train_set['date_block_num'] ==row-1].values[:10000,:])
    for row in range(2, 13):
        val_y_set.append(train_set[train_set['date_block_num'] == row].values[:10000, -1])
        val_x_set.append(train_set[train_set['date_block_num'] == row-1].values[:10000,:])

    return np.array(train_x_set, dtype=np.float32), np.array(train_y_set, dtype=np.float32), np.array(val_x_set, dtype=np.float32), np.array(
        val_y_set, dtype=np.float32), test_x_set, test_y_set

def standartise(arr):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(arr.reshape(-1, 1))
    return scaled_df


def preprocess(parentPath):
    sales_df = pd.read_csv(parentPath + '/sales_train_v2.csv', sep=',')
    items_df = pd.read_csv(parentPath + '/items.csv', sep=',')
    item_id_arr = ['item_id0', 'item_id1', 'item_id2', 'item_id3', 'item_id4', 'item_id5', 'item_id6', 'item_id7',
                   'item_id8',
                   'item_id9', 'item_id10', 'item_id11', 'item_id12', 'item_id13', 'item_id14', 'item_id15']
    preprocessed_items_df = pd.read_csv(parentPath + '/../items2.csv', sep=',',
                                        names=item_id_arr)
    shop_id_arr = ['shop_id0', 'shop_id1', 'shop_id2', 'shop_id3', 'shop_id4', 'shop_id5', 'shop_id6']
    preprocessed_shops_df = pd.read_csv(parentPath + '/../shopIdBase2.csv', sep=',',
                                        names=shop_id_arr)
    ittt = items_df['item_category_id'].values
    itemValues = ittt[sales_df['item_id'].values]
    scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sales_df['item_price'] = scaler.fit_transform(sales_df['item_price'])
    sales_df.insert(loc=4, column='item_category', value=itemValues)
    sales_df = pd.concat([sales_df, pd.DataFrame(columns=item_id_arr)], sort=False)
    sales_df = pd.concat([sales_df, pd.DataFrame(columns=shop_id_arr)], sort=False)
    for i in range(16):
        str_i_ = 'item_id' + str(i)
        sales_df[str_i_] = preprocessed_items_df[str_i_]
    for i in range(7):
        str_i_ = 'shop_id' + str(i)
        sales_df[str_i_] = preprocessed_shops_df[str_i_]
    sales_df = sales_df.drop(columns=["shop_id", "item_id"])
    months = []
    for row in sales_df['date']:
        splitted_value = row.split(".")
        month = (int)(splitted_value[1])
        months.append(month)
    sales_df.insert(loc=0, column='month', value=months)
    sales_df = sales_df.drop(columns=['date'])
    columns = ['date_block_num', 'month', 'item_category', 'item_price'] + item_id_arr + shop_id_arr
    sales_df = sales_df.groupby(columns).sum().reset_index()
    sales_df = sales_df.rename(index=str, columns={"item_cnt_day": "item_cnt_month"})
    sales_df = sales_df.sort_values(by=columns)

    sales_df['item_price'] = standartise(sales_df['item_price'].values)
    sales = sales_df.values
    return sales


def preprocess2(parentPath):
    sales_df = pd.read_csv(parentPath + '/sales_train_v2.csv', sep=',')
    items_df = pd.read_csv(parentPath + '/items.csv', sep=',')
    ittt = items_df['item_category_id'].values
    itemValues = ittt[sales_df['item_id'].values]
    sales_df.insert(loc=4, column='item_category', value=itemValues)

    sales_df = sales_df.drop(columns=['date'])
    sales_df = sales_df.sort_values(["date_block_num", "item_id", "shop_id", "item_category"]).reset_index()
    sale_count_df = sales_df[["date_block_num", "item_id", "shop_id", "item_category", "item_cnt_day"]]
    sale_count_df = sale_count_df.groupby(["date_block_num", "item_id", "shop_id", "item_category"]).sum().reset_index()
    sale_count_df = pd.pivot_table(sale_count_df, values='item_cnt_day', index=['shop_id', 'item_id'],
                                   columns=['date_block_num'],
                                   fill_value=0).values

    return sale_count_df


def getRows(data_block, itemInd, shopInd, sales_df):
    return sales_df[sales_df.item_id.isin([itemInd]) & sales_df.date_block_num.isin([data_block]) & sales_df.shop_id.isin([shopInd]) ]


def preprocess3(parentPath):
    sales_df = pd.read_csv(parentPath + '/../son.csv', sep=',')
    # sales_df = pd.read_csv(parentPath + '/sales_train_v2.csv', sep=',')
    items_df = pd.read_csv(parentPath + '/items.csv', sep=',')
    ittt = items_df['item_category_id'].values
    itemValues = ittt[sales_df['item_id'].values]
    sales_df.insert(loc=1, column='item_category', value=itemValues)
    scaler = MinMaxScaler(feature_range=(-1,1))
    # sales_df['item_cnt_day']=scaler.fit_transform(sales_df['item_cnt_day'].values.reshape(-1, 1))
    sales_df['month']=sales_df['date_block_num']%12
    sales_df['shop_id']=scaler.fit_transform(sales_df['shop_id'].values.reshape(-1, 1))
    sales_df['shop_id'] = scaler.fit_transform(sales_df['shop_id'].values.reshape(-1, 1))
    sales_df = sales_df[['date_block_num','month','item_id','shop_id','item_category','item_cnt_day']]

    return sales_df

def nominalToNumeric(keyArr, valueArr):
    frameDict = {}
    for i in range(len((keyArr))):
        frameDict[keyArr[i]] = valueArr[i]
    nominal = pd.DataFrame(frameDict)
    baseEncoder = ce.BaseNEncoder(cols=keyArr)
    return baseEncoder.fit_transform(nominal)
