from datetime import datetime
from operator import itemgetter
import load_db_data
from load_db_data import maketable
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from tqdm import tqdm

# dataframe에서 db로 저장
db_address = 'mysql+pymysql://vision:vision9551@183.111.103.165/kisti_crawl_test'
db_conn = create_engine(db_address)
conn = db_conn.connect()

except_list = list()
separator = " ~ "
create_date = str(datetime.now()).split(' ')[0].strip()
URL_LIST = open("url_input.txt","r",encoding="utf-8").read().splitlines()

# dunn index
def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def make_cluster_table(df):
    df = df.dropna(subset=['product_price'], how='any', axis=0)
    
    # 2차원(가격X가격) 설정 및 데이터정규화
    price = pd.DataFrame(df,columns = ['product_price', 'product_price'])

    scaler = MinMaxScaler()
    data_scale = scaler.fit_transform(price)

    # dunn index 계산(최대 가격 클러스터 수를 5로 설정)
    dunn_list = []
    for i in range(2, 6) :
        clus = []
        np.random.seed(100)
        kmeans = KMeans(n_clusters = i)
        y_pred = kmeans.fit_predict(data_scale)
        price['labels'] = kmeans.labels_
        
        for j in range(i) :
            clus.append(price[price['labels'] == j].values)
        dunn_list.append(dunn(clus))

    k = np.argmax(dunn_list) + 2
    # print(f'최적 클러스터 수 : {k}개')

     # K-means 클러스터링
    model = KMeans(n_clusters=k, random_state=10)
    model.fit(data_scale)
    df['cluster'] = model.fit_predict(data_scale)

    # 가격 범위
    for j in range(k):
        cluster_price = df.loc[df['cluster'] == j, 'product_price']
        cluster_name_dict = dict()
        cluster_name_dict["price_range"] = f"{min(cluster_price)};{max(cluster_price)}"
        cluster_name_dict["price_min"] = int(min(cluster_price))
        cluster_name_dict["cluster"] = j
        cluster_name_dict_list.append(cluster_name_dict)
    return df

def quality_table(url,cluster_quality):
    # amz_cluster_dataframe
    cluster_product_info = cluster_product_info_merge_price_range
    # print(cluster_product_info)
    
    # quality title list를 가져오는 부분
    quality_title_url = maketable(url)
    quality_title_list = quality_title_url.quality_title(cluster_quality)
    
    # print(quality_title_list)

    # quality title과 content를 dict로 가져오는 부분
    content_list = quality_title_url.quality_content(quality_title_list,cluster_quality)
    content_list = list(map(dict, set(tuple(sorted(d.items())) for d in content_list)))
    # content_list = list(set(content_list))

    # quality title마다 dataframe을 만들어서 cluster_product_info에 merge하는 부분
    quality_index = 1
    
    for quality_title in quality_title_list:
        # print(quality_title)
        quality_title_content_list = list()
        for content in content_list:
            # print(quality_title)
            if quality_title == content["feature_title"]:
                quality_title_content_list.append(content)
        quality_df = pd.DataFrame(quality_title_content_list)
        
        quality_df.rename(columns = {'feature_title':f'title{quality_index}','feature_rating':f'rating{quality_index}'},inplace=True)
        quality_index += 1
        quality_df = quality_df.drop_duplicates()
        # print(quality_df)
        
        cluster_product_info = pd.merge(left = cluster_product_info , right = quality_df, how = "left", on = "product_key")
        # print(cluster_product_info)

    return cluster_product_info

# 가격 범위
def rating_min_max():
    #초기 temp 선언
    temp = pd.DataFrame({'url' : url ,'cluster' : list(range(cluster_length))})
    #품질특성 title 수만큼 반복
    
    for num in range(len(quality_title_list)):
        tb_list = list()
        title_list = amz_product_info_df.loc[:, [f'title{num+1}']].values.tolist()
        title_list = sum(title_list,[])
        title_list = [i for i in title_list if i != None and pd.isnull(i)==False]
        title = title_list[0]
        #클러스터 수만큼 반복
        for cluster_index in range(cluster_length) :
            content_list = amz_product_info_df.loc[amz_product_info_df['cluster'] == cluster_index, f'rating{num+1}']
            content_list = [i for i in content_list if i != None and pd.isnull(i)==False]
            s_range = [cluster_index, title, f"{min(content_list, default = None)}{separator}{max(content_list, default = None)}"]
            tb_list.append(s_range)
        rating_df = pd.DataFrame(tb_list, columns = ['cluster', f'title{num+1}', f'rating{num+1}'])
        temp = pd.merge(left = temp , right = rating_df, how = "left", on = "cluster")
    return temp


for url in tqdm(URL_LIST):
    try:
        cluster_name_dict_list = list()
        product_info = load_db_data.amz_product_info_dataframe(url)
        # print(product_info)

        try:
            #cluster계산하여 저장 (df형태)
            cluster_product_info = make_cluster_table(product_info)
        except:
            product_info['cluster'] = 0
            cluster_product_info = product_info
            cluster_name_dict_list = [{
                'price_range':'',
                'price_min':'',
                'cluster':0
            }]

        #cluster_name_dict_list 가격순위로 정렬
        cluster_name_dict_list = sorted(cluster_name_dict_list,key=itemgetter('price_min'))
        # print(cluster_name_dict_list)

        #cluster별 price_range 계산하여 df로 저장하기
        cluster_price_range_list = list()


        for i in range(len(cluster_name_dict_list)):
            price_name_list = list()
            cluster_price_range_dict = dict()
            cluster_price_range_dict['cluster'] = cluster_name_dict_list[i]['cluster']
            cluster_price_range_dict['price_range'] = cluster_name_dict_list[i]['price_range']
            if len(cluster_name_dict_list) == 1:
                if cluster_name_dict_list[i]['price_range']== '':
                    price_name_list = ['불능클러스터']
                else:
                    price_name_list = ['단일클러스터']
            if len(cluster_name_dict_list) == 2:
                price_name_list = ['저가','고가']
            if len(cluster_name_dict_list) == 3:
                price_name_list = ['저가','중가','고가']
            if len(cluster_name_dict_list) == 4:
                price_name_list = ['저가','중저가','중고가','고가']
            if len(cluster_name_dict_list) == 5:
                price_name_list = ['저가','중저가','중가','중고가','고가']
            cluster_price_range_dict['cluster_name'] = price_name_list[i]
            cluster_price_range_list.append(cluster_price_range_dict)
        cluster_price_range_df = pd.DataFrame(cluster_price_range_list)
        # print(cluster_price_range_df)


        # #cluster_product_info와 price_range의 merge 저장
        cluster_product_info_merge_price_range = pd.merge(left = cluster_product_info , right = cluster_price_range_df, how = "left", on = "cluster")
        # cluster_product_info_merge_price_range.to_sql(name='amz_cluster_tb',con=db_conn,if_exists='append',index=False)

        #품질특성 merge하여 저장
        product_feature_rating = load_db_data.amz_feature_rating_dataframe(url)
        cluster_quality = pd.merge(left = cluster_product_info_merge_price_range , right = product_feature_rating, how = "left", on = "product_key")
        # cluster_quality.to_sql(name='amz_cluster_quality_tb',con=db_conn,if_exists='append',index=False)


        # 품질특성 title 추출
        quality_title_url = maketable(url)
        quality_title_list = quality_title_url.quality_title(cluster_quality)
        # print(quality_title_list)



        quality_tb = quality_table(url,cluster_quality)
        quality_tb["title_list"] = str(quality_title_list)
        quality_tb["create_date"] = create_date
        quality_tb.to_sql(name='tb_amz_model4_data',con=db_conn,if_exists='append',index_label="idx",index=False)




        amz_product_info_df = pd.DataFrame(quality_tb,columns =['url','cluster','price_range','product_key','product_name','product_price','title1','rating1','title2','rating2','title3','rating3','title4','rating4','title5','rating5','title6','rating6','title7','rating7'])
        amz_product_info_df = amz_product_info_df.loc[:,['cluster','title1','rating1','title2','rating2','title3','rating3','title4','rating4','title5','rating5','title6','rating6','title7','rating7']]

        #클러스터링 길이를 추출하는 수학식
        cluster_length = len(amz_product_info_df.drop_duplicates(["cluster"]))


        amz_quality_tb = rating_min_max()

        result_df = pd.merge(left = amz_quality_tb, right = cluster_price_range_df, how = 'left', on = 'cluster')
        result_df["title_list"] = str(quality_title_list)
        result_df["create_date"] = create_date
        result_df.to_sql(name='tb_amz_model4',con=db_conn,if_exists='append',index_label="idx",index=False)
        print(except_list)
    except:
        except_list.append(url)
        print(except_list)