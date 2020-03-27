#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import svm

import collections
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict


import warnings

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import HiveContext
import json
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import count
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import approx_count_distinct
from pyspark.sql.functions import first, last
from pyspark.sql.functions import min, max
from pyspark.sql.functions import sum, count, avg, expr, mean
from pyspark.sql.functions import var_pop, stddev_pop
from pyspark.sql.functions import var_samp, stddev_samp
from pyspark.sql.functions import skewness, kurtosis

from pyspark.sql.functions import var_pop, stddev_pop
from pyspark.sql.functions import var_samp, stddev_samp

from pyspark.sql.functions import corr, covar_pop, covar_samp
from pyspark.sql.functions import collect_set, collect_list

from pyspark.sql.functions import col, to_date

import datetime

from pyspark.sql.window import Window
from pyspark.sql.functions import desc
from pyspark.sql.functions import max
from pyspark.sql.functions import sum

from pyspark.sql.functions import dense_rank, rank
from pyspark.sql.functions import col


from pyspark.sql.types import DataType
from pyspark.sql.types import NullType
from pyspark.sql.types import StringType
from pyspark.sql.types import BinaryType
from pyspark.sql.types import BooleanType
from pyspark.sql.types import DateType

from pyspark.sql.types import TimestampType
from pyspark.sql.types import DecimalType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import ByteType
from pyspark.sql.types import IntegerType

from pyspark.sql.types import LongType
from pyspark.sql.types import ShortType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import MapType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

from pyspark.ml.fpm import FPGrowth


import copy
warnings.filterwarnings("ignore")


# In[86]:


csvFile = "./OLD_output.csv"


# In[87]:


pandasDf = pd.read_csv(csvFile)


# In[88]:


pandasDf


# In[89]:


pandasDf.drop(inplace=True, axis=1, columns=['Unnamed: 0'])


# In[90]:


pandasDf


# In[91]:


pandasDf.columns


# In[92]:


cols = pandasDf.columns


# In[93]:


cols_set = set()
for i, val in enumerate(cols):
    cols_set.add(val.split(';')[0])


# In[94]:


len(cols_set)


# In[95]:


pandasDf['items'] = ''


# In[96]:


pandasDf.dtypes


# In[97]:


for i, row in pandasDf.iterrows():
    item_list = []
    for col in pandasDf.columns:
        if(row[col] == True):
            for c in cols_set:
                if (col.find(c) != -1):
                    item_list.append(c)
    pandasDf.at[i, 'items'] = item_list


# In[98]:


pandasDf['items'].unique


# In[99]:


for col in pandasDf.columns:
    if col != 'items':
        pandasDf[[col]]=pandasDf[[col]].astype('int')


# In[100]:


pandasDf


# In[101]:


for i, row in pandasDf.iterrows():
    a = []
    if not row['items']:
        print('list is empty')


# In[102]:


itemsDf = pandasDf['items']


# In[103]:


itemsDf


# In[104]:


spark = SparkSession     .builder     .appName("Python Spark create RDD example")     .getOrCreate()


# In[105]:


df = spark.createDataFrame(pandasDf)


# In[106]:


df.select("items").show()


# In[107]:


fpGrowth = FPGrowth(itemsCol="items", minSupport=0.1, minConfidence=0.2)


# In[108]:


model = fpGrowth.fit(df)


# In[ ]:


# Display frequent itemsets.
model.freqItemsets.show()


# In[ ]:


# Display generated association rules.
model.associationRules.show()


# In[ ]:


# Display generated association rules.
model.associationRules.show()


# In[ ]:





# In[ ]:




