#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import random
import time
from datetime import timedelta, datetime
import re
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import argparse
import pickle
import ast
import gc
from dataset.aliccp.preprocess_ali_ccp import reduce_mem


class DataPreprocessing(object):
    def __init__(self, data_path, dataset_name, domains=[], k_cores=3, prepare2train_month=6,
                 downsample_freq_thresh=10, sample_n_domain=50,
                 sample_mode="mix_interval_random", discrete_method="uniform"):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.domains = domains
        self.k_cores = k_cores
        self.downsample_freq_thresh = downsample_freq_thresh
        self.sample_n_domain = sample_n_domain
        self.sample_mode = sample_mode
        self.discrete_method = discrete_method
        if dataset_name == 'amazon':
            self.feature_names = ['userid', 'itemid', 'weekday', 'domain',
                                  'sales_chart', 'sales_rank', 'brand', 'price']
            # 按照domain中数据量大小进行排序
            self.domain2encoder_dict = {'Clothing, Shoes & Jewelry': 0, 'Home & Kitchen': 1, 'Books': 2,
                                        'Electronics': 3, 'Sports & Outdoors': 4, 'Tools & Home Improvement': 5,
                                        'Pet Supplies': 6, 'Automotive': 7, 'Grocery & Gourmet Food': 8,
                                        'Patio, Lawn & Garden': 9, 'Office Products': 10, 'Toys & Games': 11,
                                        'Cell Phones & Accessories': 12, 'Movies & TV': 13, 'Arts, Crafts & Sewing': 14,
                                        'Industrial & Scientific': 15, 'Kindle Store': 16, 'Musical Instruments': 17,
                                        'Appliances': 18, 'CDs & Vinyl': 19, 'Video Games': 20, 'Gift Cards': 21,
                                        'Magazine Subscriptions': 22, 'Home & Business Services': 23,
                                        'Collectibles & Fine Art': 24}
            self.feature_dims = None
            self.prepare2train_month = prepare2train_month
            self.preprocess_path = os.path.join(self.data_path, f'prepare2train_filter_{self.prepare2train_month}month.csv')
            self.label_name = 'label'
        elif dataset_name == 'aliccp':
            categorical_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207',
                                   '210', '216', '508', '509', '702', '853', '109_14', '110_14', '127_14', '150_14',
                                   '301']
            numerical_columns = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
            self.feature_names = categorical_columns + numerical_columns
            self.domain2encoder_dict = {str(item): item for item in range(self.sample_n_domain)}
            self.preprocess_path = os.path.join(self.data_path,
                                                f'thresh{self.downsample_freq_thresh}_ndomain{self.sample_n_domain}_mode{self.sample_mode}.csv')
            self.label_name = 'click'

        self.one_hot_feature_names = [f for f in self.feature_names if 'seq' not in f]
        self.feature_dims, self.itemid_all = None, None

    # for amazon
    @staticmethod
    def process_price(price_str):
        try:
            if not isinstance(price_str, str) or pd.isnull(price_str) or price_str == '':
                return None
            cleaned_price = re.sub('[^\d.-]', '', price_str)
            if '-' in cleaned_price:
                prices = cleaned_price.split('-')
                price = np.mean([float(p) for p in prices])
            else:
                price = float(cleaned_price)
            return np.ceil(price)
        except ValueError:
            return None

    @staticmethod
    def process_rank(sales_rank_str):
        if not isinstance(sales_rank_str, str):
            return None, None
        try:
            rank_part, chart_part = sales_rank_str.split(' in ')
            rank = int(rank_part.replace(',', ''))
            chart = chart_part.split(' (')[0]
            return rank, chart
        except ValueError:
            return None, None

    @staticmethod
    def extract_pos_neg_seq(row):
        positive_item_seq = []
        positive_item_seq_timestamp = []
        negative_item_seq = []
        negative_item_seq_timestamp = []

        for item, label, timestamp in zip(row['item_seq'], row['item_seq_label'], row['item_seq_timestamp']):
            if label == 1:
                positive_item_seq.append(item)
                positive_item_seq_timestamp.append(timestamp)
            elif label == 0:
                negative_item_seq.append(item)
                negative_item_seq_timestamp.append(timestamp)

        return positive_item_seq, positive_item_seq_timestamp, negative_item_seq, negative_item_seq_timestamp

    @staticmethod
    def aggregate_pos_neg_seq(group):
        pos_items = group.loc[group['label'] == 1, 'itemid'].tolist()
        pos_timestamps = group.loc[group['label'] == 1, 'timestamp'].tolist()
        neg_items = group.loc[group['label'] == 0, 'itemid'].tolist()
        neg_timestamps = group.loc[group['label'] == 0, 'timestamp'].tolist()

        return pd.Series({
            'pos_item_seq': pos_items,
            'pos_item_seq_timestamp': pos_timestamps,
            'neg_item_seq': neg_items,
            'neg_item_seq_timestamp': neg_timestamps
        })

    # for amazon
    def merge_metadata(self, df, k_cores):
        metadata_path = os.path.join(self.data_path, 'All_Amazon_Meta.json')

        # k-cores filter
        print('before k-cores filter: df shape = ', df.shape)
        df['user_count'] = df.groupby('userid')['userid'].transform('count')
        df['item_count'] = df.groupby('itemid')['itemid'].transform('count')
        if k_cores > 0:
            df = df.loc[df.user_count >= k_cores]
            df = df.loc[df.item_count >= k_cores].copy()
        unique_items = set(df.itemid.unique())
        nunique_items = df.itemid.nunique()
        print(f'after k-cores filter: df shape = {df.shape}')
        print(f'user unique = {df.userid.nunique()}, item unique = {df.itemid.nunique()}')

        # read item metadata
        item_meta_df_path = os.path.join(self.data_path, f'item_meta_{self.k_cores}cores_{self.prepare2train_month}month.csv')
        if os.path.exists(item_meta_df_path):
            item_meta_df = pd.read_csv(item_meta_df_path)
        else:
            item_meta_df = list()
            item_cnt = 0
            with open(metadata_path, 'rb') as f:
                tqdm_bar = tqdm(f, smoothing=0, mininterval=100.0)
                for line in tqdm_bar:
                    line = json.loads(line)
                    if line['asin'] not in unique_items:
                        continue
                    item_meta_df.append([line['asin'], line['price'], line['rank'], line['brand'], line['category']])

                    item_cnt += 1
                    if item_cnt % 1000 == 0:
                        tqdm_bar.set_description(f"Processed {item_cnt}/{nunique_items} items")

                    if item_cnt >= nunique_items:
                        break
            item_meta_df = pd.DataFrame(item_meta_df, columns=['itemid', 'price', 'salesRank', 'brand', 'category'])
            item_meta_df.to_csv(item_meta_df_path, index=False)
        print(f'item_meta_df shape is {item_meta_df.shape}')

        # process item meta data
        item_meta_df.replace('', None, inplace=True)
        item_meta_df['price'] = item_meta_df['price'].apply(self.process_price)
        item_meta_df['sales_rank'], item_meta_df['sales_chart'] = zip(*item_meta_df['salesRank'].apply(self.process_rank))
        item_meta_df['tags'] = item_meta_df['category'].apply(ast.literal_eval)
        item_meta_df['domain'] = item_meta_df['tags'].apply(lambda x: x[0] if isinstance(x, list)
                                                                              and len(x) > 0 else None)
        brand_counts = item_meta_df['brand'].value_counts()
        brands_to_replace = brand_counts[brand_counts < 10].index
        item_meta_df['brand'] = item_meta_df['brand'].apply(lambda x: None if x in brands_to_replace else x)

        # process label
        label_threshold = 4.0
        df['label'] = 0
        df.loc[(df.rating > label_threshold), 'label'] = 1

        # encoder itemid
        lbe = LabelEncoder()
        lbe.fit(list(unique_items))
        df['itemid'] = lbe.transform(df['itemid'].astype(str))
        item_meta_df['itemid'] = lbe.transform(item_meta_df['itemid'].astype(str))
        # with open(os.path.join(self.data_path, 'itemid_encoder.pkl'), 'wb') as f:
        #     pickle.dump(lbe, f)

        # process user metadata
        """
        start = time.time()
        df.sort_values('timestamp', inplace=True, ignore_index=True)
        pos_df, neg_df = df.loc[df.label == 1].copy(), df.loc[df.label == 0].copy()
        pos_user_meta_df = pos_df.groupby('userid').agg({
            'itemid': lambda x: list(x),
            'timestamp': lambda x: list(x)
        }).reset_index().rename(columns={'itemid': 'pos_item_seq', 'timestamp': 'pos_item_seq_timestamp'})
        neg_user_meta_df = neg_df.groupby('userid').agg({
            'itemid': lambda x: list(x),
            'timestamp': lambda x: list(x)
        }).reset_index().rename(columns={'itemid': 'neg_item_seq', 'timestamp': 'neg_item_seq_timestamp'})
        print(f'pos_user_meta_df shape = {pos_user_meta_df.shape}, neg_user_meta_df shape = {neg_user_meta_df.shape}')
        user_meta_df = pd.merge(pos_user_meta_df, neg_user_meta_df, on='userid', how='outer')
        user_meta_df['pos_item_seq'] = user_meta_df['pos_item_seq'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['pos_item_seq_timestamp'] = user_meta_df['pos_item_seq_timestamp'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['neg_item_seq'] = user_meta_df['neg_item_seq'].apply(lambda x: x if isinstance(x, list) else [])
        user_meta_df['neg_item_seq_timestamp'] = user_meta_df['neg_item_seq_timestamp'].apply(lambda x: x if isinstance(x, list) else [])
        end = time.time()
        print(f'user_meta_df shape = {user_meta_df.shape}, build time = {end - start:.2f}s')
        

        # merge user_meta_df to df
        start = time.time()
        df.sort_values('userid', inplace=True, ignore_index=True)
        user_meta_df.sort_values('userid', inplace=True, ignore_index=True)
        df['df2user_meta_df'] = user_meta_df['userid'].searchsorted(df['userid'], side='left')

        def get_user_items_seq(row, user_meta_df, delta_days, is_pos):
            user_meta_row = user_meta_df.iloc[row['df2user_meta_df']]
            start_time = row['timestamp'] - delta_days
            end_time = row['timestamp']
            item_seq = user_meta_row['pos_item_seq'] if is_pos else user_meta_row['neg_item_seq']
            item_seq_timestamp = user_meta_row['pos_item_seq_timestamp'] if is_pos \
                else user_meta_row['neg_item_seq_timestamp']
            selected_items = [item for item, timestamp in zip(item_seq, item_seq_timestamp) if
                              start_time <= timestamp < end_time]
            return selected_items

        m = 6
        days_n = 30*m
        delta_days = int(timedelta(days=days_n - 1).total_seconds())
        df[f'user_pos_{m}month_seq'] = df.apply(get_user_items_seq, axis=1,
                                                user_meta_df=user_meta_df, delta_days=delta_days, is_pos=True)
        df[f'user_neg_{m}month_seq'] = df.apply(get_user_items_seq, axis=1,
                                                user_meta_df=user_meta_df, delta_days=delta_days, is_pos=False)
        print(f'finish getting df.user_pos/neg_{m}month_seq')
        end = time.time()
        print(f'df shape = {df.shape}, get df.item_seq time = {end - start:.2f}s')
        """

        df = df.merge(item_meta_df, on='itemid', how='left')
        print('finish merge item meta data to df')
        # item_meta_df.set_index('itemid', inplace=True).to_dict('index')
        #
        # for col in ['price', 'sales_rank', 'sales_chart', 'brand', 'domain']:
        #     df[col] = df['itemid'].map(lambda x: item_meta_df.get(x, {}).get(col, np.nan))

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['weekday'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour  # hour都是0

        return df

    # for ali-ccp
    def discrete(self, discrete_paths):
        print("Discretize continuous features, fit and transform on train, transform on val and test")
        print(discrete_paths)
        # train_path, val_path, test_path
        train_val_test_path = (os.path.join(self.data_path, 'ali_ccp_train.csv'),
                               os.path.join(self.data_path, 'ali_ccp_val.csv'),
                               os.path.join(self.data_path, 'ali_ccp_test.csv'))
        if not all([os.path.exists(path) for path in train_val_test_path]):
            raise ValueError("Train, val, test data not prepared. Please run preprocess_ali_ccp.py first")
        else:
            print("Train, val, test data already prepared")
        # train_df, val_df, test_df = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
        train_val_test_df = (pd.read_csv(train_val_test_path[0]),
                             pd.read_csv(train_val_test_path[1]),
                             pd.read_csv(train_val_test_path[2]))
        print("train_val_test_df:", [df.shape for df in train_val_test_df])

        from sklearn.preprocessing import KBinsDiscretizer
        # combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        columns_to_discretize = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
        print("columns_to_discretize:", columns_to_discretize)

        # 使用KBinsDiscretizer进行离散化
        for column in tqdm(columns_to_discretize, mininterval=5):
            discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal',
                                           strategy=self.discrete_method,
                                           subsample=int(2e5) if self.discrete_method == 'quantile' else None)
            # combined_df[column] = discretizer.fit_transform(combined_df[[column]]).astype(int)
            discretizer.fit(train_val_test_df[0][[column]])  # 仅使用训练集来fit
            for i in range(3):
                train_val_test_df[i][column] = discretizer.transform(train_val_test_df[i][[column]]).astype(int)

        for i in range(3):
            train_val_test_df[i].rename(columns={'101': 'userid', '205': 'itemid', '206': 'domain'},
                                        inplace=True)
            train_val_test_df[i].to_csv(discrete_paths[i], index=False)
        print("Discretization done")

    def filter_dataframe_by_threshold(self, df_paths, thresh, n_domain, sample_mode):
        with open(f"{self.preprocess_path.split(',')[0]}.log", 'w') as log_file:
            df_num = len(df_paths)
            train_tags = [0, 1, 2]
            dfs, df_row_nums = [], []
            for i in range(df_num):
                dfs.append(reduce_mem(pd.read_csv(df_paths[i])))
                dfs[i]['train_tag'] = train_tags[i]  # 增加一个标记列来区分train, val, test
                df_row_nums.append(dfs[i].shape[0])
            df = pd.concat(dfs, ignore_index=True)

            import sys
            sys.stdout = log_file

            print('Columns:', df.columns)
            print('Train_tag:', train_tags[:df_num])
            print(f"Concat {df_num} dataframes to filter, original row num: {df_row_nums}")

            # 计算用户和商品的频次
            user_counts = df['userid'].value_counts()
            item_counts = df['itemid'].value_counts()

            # 过滤出现频次高于等于阈值的用户和商品
            valid_users = user_counts[user_counts >= thresh].index
            valid_items = item_counts[item_counts >= thresh].index
            valid_mask = df['userid'].isin(valid_users) & df['itemid'].isin(valid_items)

            # 应用过滤条件并得到新的DataFrame
            print("Before filter user and item:", df.shape[0])
            filtered_df = df[valid_mask]
            print("After filter user and item:", filtered_df.shape[0])

            # Filtering based on userid and itemid counts within each domain
            print("Before filter domain:", filtered_df["domain"].value_counts())
            filtered_df = filtered_df.groupby('domain').filter(
                lambda x: (x['userid'].nunique() >= thresh * 20) & (x['itemid'].nunique() >= thresh * 20))
            sort_by_count = filtered_df["domain"].value_counts().sort_values(ascending=False)
            print("After filter domain:", sort_by_count)
            print("domain counts describe:",
                  sort_by_count.describe(percentiles=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

            if sample_mode == "nlargest":
                selected_domains = sort_by_count.nlargest(n_domain).index
            elif sample_mode == "random":
                # Randomly selecting n_domain domains
                remaining_domains = set(filtered_df['domain'].unique())
                selected_domains = random.sample(remaining_domains, min(n_domain, len(remaining_domains)))
            elif sample_mode == "interval":
                # Sort domains based on count and select n evenly spaced domains
                sorted_domains = sort_by_count.index
                step = max(1, len(sorted_domains) // n_domain)
                selected_domains = sorted_domains[::step][:n_domain]
            elif sample_mode == "weighted":
                # Calculate weights based on log-transformed domain count
                domain_counts = filtered_df["domain"].value_counts()
                mid = domain_counts.median()
                domain_counts_f = (domain_counts + 0.2 * mid ** 2 / domain_counts) ** 0.8
                weights = domain_counts_f / domain_counts_f.sum()
                print("weights:", weights)
                selected_domains = np.random.choice(domain_counts.index, n_domain, p=weights, replace=False)
            elif sample_mode == "interval_random":  # 分层抽样
                # Sort domains based on count and select n domains from each interval
                sorted_domains = sort_by_count.index
                large_domains = sorted_domains[:int(0.05 * len(sorted_domains))]
                small_domains = sorted_domains[int(0.05 * len(sorted_domains)):]

                selected_domains = []
                for tmp_n_domains, tmp_sorted_domains in zip([5, n_domain - 5], [large_domains, small_domains]):
                    step = max(1, len(tmp_sorted_domains) // tmp_n_domains)
                    selected_domains.extend(tmp_sorted_domains[::step][:tmp_n_domains])
            elif sample_mode == "mix_interval_random":
                # 部分domain是由多个domain合成的，最后再分层抽样出来n_domain个
                n_mix_domain = int(1.2 * n_domain)
                sorted_domains = sort_by_count.index
                large_domains = sorted_domains[:int(0.05 * len(sorted_domains))]
                small_domains = sorted_domains[int(0.05 * len(sorted_domains)):]

                tmp_selected_domains = []
                for tmp_n_domains, tmp_sorted_domains in zip([8, n_mix_domain - 8], [large_domains, small_domains]):
                    step = max(1, len(tmp_sorted_domains) // tmp_n_domains)
                    tmp_selected_domains.extend(tmp_sorted_domains[::step][:tmp_n_domains])

                # 随机选择n_mix_domain-n_domain个domain，将其替换成n_domain中的domain，保证最后的domain数目为n_domain
                selected_domains = random.sample(tmp_selected_domains, n_domain)
                mix_source_domains = set(tmp_selected_domains) - set(selected_domains)
                mix_target_domains = random.sample(selected_domains, len(mix_source_domains))
                mix_dict = dict(zip(mix_source_domains, mix_target_domains))
                print("mix_dict from domain to:", mix_dict)
                filtered_df['domain'] = filtered_df['domain'].replace(mix_dict)
            else:
                raise ValueError("Invalid sample_mode")

            print("sample_mode:", sample_mode)
            print("selected_domains:", selected_domains)
            filtered_df = filtered_df[filtered_df['domain'].isin(selected_domains)]
            print("After select domain with sample_mode:")
            print("After final sample domain 1:", filtered_df["domain"].value_counts())

            # Mapping domains to continuous IDs
            sorted_domains_from_large = filtered_df["domain"].value_counts().sort_values(ascending=False).index.tolist()
            domain_id_mapping = {domain: i for i, domain in enumerate(sorted_domains_from_large)}
            domain_id_mapping_str = {str(domain): i for i, domain in enumerate(sorted_domains_from_large)}
            inverse_domain_id_mapping = {i: domain for domain, i in domain_id_mapping.items()}
            self.domain2encoder_dict = domain_id_mapping_str
            filtered_df['domain'] = filtered_df['domain'].map(domain_id_mapping)

            # 因为sample完了，可能少了一些user和item，因此可以重新编码
            print("Re-encoding userid and itemid after domain sampling")
            print(f"Before re-encoding, userid max: {filtered_df['userid'].max()}, "
                  f"itemid max: {filtered_df['itemid'].max()}")
            for fea in ['userid', 'itemid']:
                lbe = LabelEncoder()
                filtered_df[fea] = lbe.fit_transform(filtered_df[fea])
            print(f"After re-encoding, userid max: {filtered_df['userid'].max()}, "
                  f"itemid max: {filtered_df['itemid'].max()}")

            print("After final sample domain 2:", filtered_df["domain"].value_counts(),
                  "len", len(filtered_df["domain"]))
            sys.stdout = sys.__stdout__
        print("After final sample domain 3:", filtered_df["domain"].value_counts(),
              "len", len(filtered_df["domain"]))

        return filtered_df, domain_id_mapping, inverse_domain_id_mapping

    def update_config(self, config):
        config.domain2encoder_dict = self.domain2encoder_dict
        config.preprocess_path = self.preprocess_path

    def main(self):
        if os.path.exists(self.preprocess_path):
            # data = pd.read_csv(preprocess_path)
            print(f'{self.preprocess_path} already prepared')
        else:
            if self.dataset_name == 'amazon':
                mergemeta_path = os.path.join(self.data_path, f'mergemeta_{self.k_cores}cores_{self.prepare2train_month}month.csv')
                if os.path.exists(mergemeta_path):
                    df = pd.read_csv(mergemeta_path)
                else:
                    csv_path = os.path.join(self.data_path, f'all_csv_files_{self.prepare2train_month}month.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path, engine='c', low_memory=False, on_bad_lines='skip')
                    else:
                        rating_csv_columns = ['itemid', 'userid', 'rating', 'timestamp']
                        df = pd.DataFrame(columns=rating_csv_columns)

                        # 存最近prepare2train_month的交互记录
                        days_n = 30 * self.prepare2train_month + self.prepare2train_month // 2
                        end_date = int(datetime(2018, 8, 15).timestamp())  # df_total['timestamp'].max()
                        start_date = end_date - int(timedelta(days=days_n).total_seconds())

                        # 分块读取和处理 CSV 文件
                        for chunk in pd.read_csv(os.path.join(self.data_path, 'all_csv_files.csv'),
                                                 chunksize=int(5e7), header=None, names=rating_csv_columns, engine='c',
                                                 low_memory=False, on_bad_lines='skip'):
                            filtered_chunk = chunk.loc[(chunk['timestamp'] >= start_date) & (chunk['timestamp'] < end_date)]
                            df = pd.concat([df, filtered_chunk], ignore_index=True)

                        df.to_csv(csv_path, index=False)
                    print(f'df total shape = {df.shape}')

                    # 合并product的meta特征
                    df = self.merge_metadata(df, k_cores=self.k_cores)
                    df.to_csv(mergemeta_path, index=False)

                print('finish loading data. start preprocessing')

                # 稠密特征离散化
                df['sales_rank'] = df['sales_rank'].fillna(df['sales_rank'].quantile()).astype(int)  # sales_rank
                sales_rank_bins = [0] + list(np.exp2(np.arange(2, 21, 2)).astype(int)) + [np.inf]
                df['sales_rank'] = pd.cut(df['sales_rank'], bins=sales_rank_bins, labels=False)

                df['price'] = df['price'].fillna(df['price'].quantile()).astype(int)  # sales_rank
                price_bins = [-1] + list(np.exp2(np.arange(1, 13, 1.2)).astype(int)) + [np.inf]
                df['price'] = pd.cut(df['price'], bins=price_bins, labels=False)
                df['timestamp'] = df['timestamp'].astype(int)

                # 定长特征数据数字化，itemid already encoded
                encoder_feature_names = [fea for fea in self.one_hot_feature_names if (fea!='itemid') and (fea!='domain')]
                df[encoder_feature_names].fillna('-1', inplace=True)
                for fea in encoder_feature_names:
                    lbe = LabelEncoder()
                    df[fea] = lbe.fit_transform(df[fea].astype(str))

                df = df.loc[df['domain'].isin(self.domains)] if len(self.domains) > 0 else df
                df = df.dropna(subset=['domain'])
                df['domain'] = df['domain'].map(self.domain2encoder_dict)

                data = df[self.feature_names+['label']+['timestamp']]  # timestamp是后续划分训练测试集需要
                data.to_csv(self.preprocess_path, index=False)
                print(f'finish preprocessing {self.preprocess_path}')
            elif self.dataset_name == 'aliccp':
                discrete_paths = (os.path.join(self.data_path, f"ali_ccp_train_discrete_{self.discrete_method}.csv"),
                                  os.path.join(self.data_path, f"ali_ccp_val_discrete_{self.discrete_method}.csv"),
                                  os.path.join(self.data_path, f"ali_ccp_test_discrete_{self.discrete_method}.csv"))

                if not all([os.path.exists(path) for path in discrete_paths]):
                    self.discrete(discrete_paths)
                else:
                    print("Discrete data already prepared")

                df, domain_id_mapping, inverse_domain_id_mapping = self.filter_dataframe_by_threshold(discrete_paths,
                                                                                                      self.downsample_freq_thresh,
                                                                                                      self.sample_n_domain,
                                                                                                      self.sample_mode)
                df.to_csv(self.preprocess_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_cores', default=3)
    parser.add_argument('--seed', type=int, default=2000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    DataPreprocessing('dataset/aliccp', 'aliccp', downsample_freq_thresh=10, sample_n_domain=50,).main()
