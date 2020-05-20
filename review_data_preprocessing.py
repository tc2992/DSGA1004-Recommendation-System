#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql.types import DateType,IntegerType, FloatType
from pyspark.sql.functions import col,lower,regexp_replace,when
from pyspark.ml.feature import CountVectorizer,Tokenizer,IDF,StopWordsRemover
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import numpy as np
import MAP


# user_id, book_id matching
review = spark.read.json('goodreads_reviews_dedup.json')
user = spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv',header=True)
book = spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv',header=True)
review_ = review.join(user,'user_id').join(book,'book_id')
review_data = review_.select(col('user_id_csv').alias('user_id'),col('book_id_csv').alias('book_id'),'rating','review_text','review_id')
review_data = review_data.withColumn('rating',review_data['rating'].cast(FloatType()))
review_data = review_data.withColumn('user_id',review_data['user_id'].cast(IntegerType()))
review_data = review_data.withColumn('book_id',review_data['book_id'].cast(IntegerType()))
review_data.write.parquet('review.parquet')


# 1% of data
train = spark.read.parquet("train01.parquet")
val = spark.read.parquet("val01.parquet")
test = spark.read.parquet("test01.parquet")
train = train.withColumn("rating",train["rating"].cast(FloatType()))
train = train.withColumn("user_id",train["user_id"].cast(IntegerType()))
train = train.withColumn("book_id",train["book_id"].cast(IntegerType()))
test = test.withColumn("rating",test["rating"].cast(FloatType()))
test = test.withColumn("user_id",test["user_id"].cast(IntegerType()))
test = test.withColumn("book_id",test["book_id"].cast(IntegerType()))
val = val.withColumn("rating",val["rating"].cast(FloatType()))
val = val.withColumn("user_id",val["user_id"].cast(IntegerType()))
val = val.withColumn("book_id",val["book_id"].cast(IntegerType()))

review = spark.read.parquet('review.parquet')
train_review = train.join(review,['user_id','book_id','rating']).select('review_text','rating','user_id','book_id')
val_review = val.join(review,['user_id','book_id','rating']).select('review_text','rating','user_id','book_id')
test_review = test.join(review,['user_id','book_id','rating']).select('review_text','rating','user_id','book_id')


# clean text data (tokenization + remove stop words)
tokenizer = Tokenizer(inputCol="review_text", outputCol="review_words")
remover = StopWordsRemover(inputCol="review_words", outputCol="filtered_words")

train_review = train_review.withColumn('review_text', regexp_replace(train_review.review_text,',|\.|:|!|"|\?|;|\(|\)|#|\$|~|&|',''))
train_review = train_review.withColumn('review_text', lower(col('review_text')))
train_review = tokenizer.transform(train_review)
train_review = remover.transform(train_review)

val_review = val_review.withColumn('review_text', regexp_replace(val_review.review_text,',|\.|:|!|"|\?|;|\(|\)|#|\$|~|&|',''))
val_review = val_review.withColumn('review_text', lower(col('review_text')))
val_review = tokenizer.transform(val_review)
val_review = remover.transform(val_review)

test_review = test_review.withColumn('review_text', regexp_replace(test_review.review_text,',|\.|:|!|"|\?|;|\(|\)|#|\$|~|&|',''))
test_review = test_review.withColumn('review_text', lower(col('review_text')))
test_review = tokenizer.transform(test_review)
test_review = remover.transform(test_review)


# CountVectorizer
cv = CountVectorizer(inputCol="filtered_words", outputCol="features")
model = cv.fit(train_review)
train_review_feature = model.transform(train_review)
val_review_feature = model.transform(val_review)
test_review_feature = model.transform(test_review)

idf = IDF(inputCol="features", outputCol="idf_features")
idfModel = idf.fit(train_review_feature)
train_review_feature = idfModel.transform(train_review_feature).select('user_id','book_id','rating','idf_features')
val_review_feature = idfModel.transform(val_review_feature).select('user_id','book_id','rating','idf_features')
test_review_feature = idfModel.transform(test_review_feature).select('user_id','book_id','rating','idf_features')


# write data
train_review_feature.write.parquet('train_review01.parquet')
val_review_feature.write.parquet('val_review01.parquet')
test_review_feature.write.parquet('test_review01.parquet')
