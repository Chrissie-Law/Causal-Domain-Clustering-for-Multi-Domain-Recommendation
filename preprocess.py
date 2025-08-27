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
    """
    Dataset preprocessing entry for both Amazon and AliCCP.

    Responsibilities:
      - Configure dataset-specific feature spaces and preprocessing targets
      - For Amazon: parse product metadata (price/rank/brand/category), discretize dense fields, encode categories
      - For AliCCP: discretize continuous features (KBinsDiscretizer), sample/merge domains, filter by frequency
      - Persist a single CSV ready for the training pipeline (used by main.py)
    """
    def __init__(self, data_path, dataset_name, domains=[], k_cores=3, prepare2train_month=6,
                 downsample_freq_thresh=10, sample_n_domain=50,
                 sample_mode="mix_interval_random", discrete_method="uniform"):
        """
        Args:
            data_path (str): Root directory containing raw/preprocessed files.
            dataset_name (str): 'amazon' or 'aliccp'.
            domains (list): Optional white-list of target domains (Amazon only).
            k_cores (int): k-core filter for (user,item) minimum interactions (Amazon).
            prepare2train_month (int): Sliding window size (months) for Amazon temporal filtering.
            downsample_freq_thresh (int): Min frequency threshold for user/item filtering (AliCCP).
            sample_n_domain (int): Target number of domains after sampling (AliCCP).
            sample_mode (str): Strategy for domain selection/merging (AliCCP).
            discrete_method (str): KBinsDiscretizer strategy for numeric features (AliCCP).
        """
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
            # Domains sorted by data volume
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
        """
        Parse noisy price strings into a single numeric bucket (ceil).
        Handles ranges like "$12.99 - $15.99" by averaging, and strips non-numeric chars.

        Args:
            price_str (str): Raw price field from metadata.

        Returns:
            float or None: Ceiled price value; None if parse fails.
        """
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
        """
        Extract item rank and chart/category from Amazon 'salesRank' string.

        Args:
            sales_rank_str (str): e.g., "1,234 in Electronics"

        Returns:
            (int or None, str or None): (rank, chart_category)
        """
        if not isinstance(sales_rank_str, str):
            return None, None
        try:
            rank_part, chart_part = sales_rank_str.split(' in ')
            rank = int(rank_part.replace(',', ''))
            chart = chart_part.split(' (')[0]
            return rank, chart
        except ValueError:
            return None, None

    # for amazon
    def merge_metadata(self, df, k_cores):
        """
        Merge Amazon product metadata into interactions and perform basic cleaning/encoding.

        Args:
            df (pd.DataFrame): Raw interactions with ['itemid','userid','rating','timestamp'].
            k_cores (int): Threshold for k-core filtering (>= k on both user and item).

        Returns:
            pd.DataFrame: Enriched interactions with metadata.
        """
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
        item_meta_df_path = os.path.join(self.data_path,
                                         f'item_meta_{self.k_cores}cores_{self.prepare2train_month}month.csv')
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

        # process label (ratings > 4 are treated as positives)
        label_threshold = 4.0
        df['label'] = 0
        df.loc[(df.rating > label_threshold), 'label'] = 1

        # encode itemid
        lbe = LabelEncoder()
        lbe.fit(list(unique_items))
        df['itemid'] = lbe.transform(df['itemid'].astype(str))
        item_meta_df['itemid'] = lbe.transform(item_meta_df['itemid'].astype(str))

        df = df.merge(item_meta_df, on='itemid', how='left')
        print('finish merge item meta data to df')

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['weekday'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour

        return df

    # for ali-ccp
    def discrete(self, discrete_paths):
        """
        Discretize continuous features for AliCCP using KBinsDiscretizer.

        Args:
            discrete_paths (tuple): (train_out_path, val_out_path, test_out_path)
        """
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
        train_val_test_df = (pd.read_csv(train_val_test_path[0]),
                             pd.read_csv(train_val_test_path[1]),
                             pd.read_csv(train_val_test_path[2]))
        print("train_val_test_df:", [df.shape for df in train_val_test_df])

        from sklearn.preprocessing import KBinsDiscretizer
        columns_to_discretize = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
        print("columns_to_discretize:", columns_to_discretize)

        # Discretize using KBinsDiscretizer (fit on train; transform val/test)
        for column in tqdm(columns_to_discretize, mininterval=5):
            discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal',
                                           strategy=self.discrete_method,
                                           subsample=int(2e5) if self.discrete_method == 'quantile' else None)
            discretizer.fit(train_val_test_df[0][[column]])  # fit on training set only
            for i in range(3):
                train_val_test_df[i][column] = discretizer.transform(train_val_test_df[i][[column]]).astype(int)

        for i in range(3):
            train_val_test_df[i].rename(columns={'101': 'userid', '205': 'itemid', '206': 'domain'},
                                        inplace=True)
            train_val_test_df[i].to_csv(discrete_paths[i], index=False)
        print("Discretization done")

    def filter_dataframe_by_threshold(self, df_paths, thresh, n_domain, sample_mode):
        """
        Filter AliCCP data by frequency and sample/merge domains according to `sample_mode`.

        Args:
            df_paths (tuple): (train_csv, val_csv, test_csv) after discretization.
            thresh (int): Frequency threshold to keep users and items.
            n_domain (int): Number of target domains to keep.
            sample_mode (str): One of {'nlargest','random','interval','weighted','interval_random','mix_interval_random'}.

        Returns:
            filtered_df (pd.DataFrame): Filtered DataFrame with sampled domains
            domain_id_mapping (dict): Mapping from original domain IDs to new IDs
            inverse_domain_id_mapping (dict): Mapping from new domain IDs to original IDs
        """
        with open(f"{self.preprocess_path.split(',')[0]}.log", 'w') as log_file:
            df_num = len(df_paths)
            train_tags = [0, 1, 2]
            dfs, df_row_nums = [], []
            for i in range(df_num):
                dfs.append(reduce_mem(pd.read_csv(df_paths[i])))
                dfs[i]['train_tag'] = train_tags[i]  # Add tag to distinguish train, val, test
                df_row_nums.append(dfs[i].shape[0])
            df = pd.concat(dfs, ignore_index=True)

            import sys
            sys.stdout = log_file

            print('Columns:', df.columns)
            print('Train_tag:', train_tags[:df_num])
            print(f"Concat {df_num} dataframes to filter, original row num: {df_row_nums}")

            # Compute user/item frequencies
            user_counts = df['userid'].value_counts()
            item_counts = df['itemid'].value_counts()

            # Keep entities with frequency >= thresh
            valid_users = user_counts[user_counts >= thresh].index
            valid_items = item_counts[item_counts >= thresh].index
            valid_mask = df['userid'].isin(valid_users) & df['itemid'].isin(valid_items)

            # Apply filtering conditions and get new DataFrame
            print("Before filter user and item:", df.shape[0])
            filtered_df = df[valid_mask]
            print("After filter user and item:", filtered_df.shape[0])

            # Filter domains by sufficient unique users/items
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
            elif sample_mode == "interval_random":  # stratified sampling
                # Sort domains based on count and select n domains from each interval
                sorted_domains = sort_by_count.index
                large_domains = sorted_domains[:int(0.05 * len(sorted_domains))]
                small_domains = sorted_domains[int(0.05 * len(sorted_domains)):]

                selected_domains = []
                for tmp_n_domains, tmp_sorted_domains in zip([5, n_domain - 5], [large_domains, small_domains]):
                    step = max(1, len(tmp_sorted_domains) // tmp_n_domains)
                    selected_domains.extend(tmp_sorted_domains[::step][:tmp_n_domains])
            elif sample_mode == "mix_interval_random":
                # Partially merge multiple domains into larger ones, then stratified sample n_domain
                n_mix_domain = int(1.2 * n_domain)
                sorted_domains = sort_by_count.index
                large_domains = sorted_domains[:int(0.05 * len(sorted_domains))]
                small_domains = sorted_domains[int(0.05 * len(sorted_domains)):]

                tmp_selected_domains = []
                for tmp_n_domains, tmp_sorted_domains in zip([8, n_mix_domain - 8], [large_domains, small_domains]):
                    step = max(1, len(tmp_sorted_domains) // tmp_n_domains)
                    tmp_selected_domains.extend(tmp_sorted_domains[::step][:tmp_n_domains])

                # Randomly select n_mix_domain-n_domain domains to replace with ones from n_domain,
                # ensuring n_domain total domains
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

            # Map domain ids to a contiguous range [0, n_selected)
            sorted_domains_from_large = filtered_df["domain"].value_counts().sort_values(ascending=False).index.tolist()
            domain_id_mapping = {domain: i for i, domain in enumerate(sorted_domains_from_large)}
            domain_id_mapping_str = {str(domain): i for i, domain in enumerate(sorted_domains_from_large)}
            inverse_domain_id_mapping = {i: domain for domain, i in domain_id_mapping.items()}
            self.domain2encoder_dict = domain_id_mapping_str
            filtered_df['domain'] = filtered_df['domain'].map(domain_id_mapping)

            # Re-encode userid/itemid after domain sampling to remove gaps and shrink id space
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
        """
        Inject preprocessing results back into the global config used by training.
        """
        config.domain2encoder_dict = self.domain2encoder_dict
        config.preprocess_path = self.preprocess_path

    def main(self):
        """
        Main preprocessing entry point. If cached preprocessed CSV exists, reuse it
        """
        if os.path.exists(self.preprocess_path):
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

                        # Keep only the most recent `prepare2train_month` interactions
                        days_n = 30 * self.prepare2train_month + self.prepare2train_month // 2
                        end_date = int(datetime(2018, 8, 15).timestamp())  # df_total['timestamp'].max()
                        start_date = end_date - int(timedelta(days=days_n).total_seconds())

                        # Chunked CSV reading and time filtering
                        for chunk in pd.read_csv(os.path.join(self.data_path, 'all_csv_files.csv'),
                                                 chunksize=int(5e7), header=None, names=rating_csv_columns, engine='c',
                                                 low_memory=False, on_bad_lines='skip'):
                            filtered_chunk = chunk.loc[(chunk['timestamp'] >= start_date) & (chunk['timestamp'] < end_date)]
                            df = pd.concat([df, filtered_chunk], ignore_index=True)

                        df.to_csv(csv_path, index=False)
                    print(f'df total shape = {df.shape}')

                    # Merge product meta features
                    df = self.merge_metadata(df, k_cores=self.k_cores)
                    df.to_csv(mergemeta_path, index=False)

                print('finish loading data. start preprocessing')

                # Discretize dense features (bucketization for rank/price)
                df['sales_rank'] = df['sales_rank'].fillna(df['sales_rank'].quantile()).astype(int)  # sales_rank
                sales_rank_bins = [0] + list(np.exp2(np.arange(2, 21, 2)).astype(int)) + [np.inf]
                df['sales_rank'] = pd.cut(df['sales_rank'], bins=sales_rank_bins, labels=False)

                df['price'] = df['price'].fillna(df['price'].quantile()).astype(int)  # sales_rank
                price_bins = [-1] + list(np.exp2(np.arange(1, 13, 1.2)).astype(int)) + [np.inf]
                df['price'] = pd.cut(df['price'], bins=price_bins, labels=False)
                df['timestamp'] = df['timestamp'].astype(int)

                # Encode fixed-length categorical features (itemid already encoded)
                encoder_feature_names = [fea for fea in self.one_hot_feature_names if (fea!='itemid') and (fea!='domain')]
                df[encoder_feature_names].fillna('-1', inplace=True)
                for fea in encoder_feature_names:
                    lbe = LabelEncoder()
                    df[fea] = lbe.fit_transform(df[fea].astype(str))

                # Optional domain filter and mapping to encoder dict
                df = df.loc[df['domain'].isin(self.domains)] if len(self.domains) > 0 else df
                df = df.dropna(subset=['domain'])
                df['domain'] = df['domain'].map(self.domain2encoder_dict)

                # Keep features + label + timestamp (timestamp used later for temporal split in main/run)
                data = df[self.feature_names+['label']+['timestamp']]  # timestamp needed for train/test split
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

    # Standalone quick preprocessing entry (AliCCP example).
    DataPreprocessing('dataset/aliccp', 'aliccp', downsample_freq_thresh=10, sample_n_domain=50,).main()
