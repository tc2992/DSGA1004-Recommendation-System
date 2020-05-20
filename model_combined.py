#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql.types import DateType,IntegerType, FloatType
from pyspark.sql import Window
from pyspark.sql.functions import col,lower,regexp_replace,when,rank,expr
from pyspark.ml.feature import CountVectorizer,Tokenizer,IDF,StopWordsRemover
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import numpy as np
import MAP
import tuning

# read data
train = spark.read.parquet("train25.parquet")
val = spark.read.parquet("val25.parquet")
test = spark.read.parquet("test25.parquet")
train = train.withColumn("rating",train["rating"].cast(FloatType()))
train = train.withColumn("user_id",train["user_id"].cast(IntegerType()))
train = train.withColumn("book_id",train["book_id"].cast(IntegerType()))
test = test.withColumn("rating",test["rating"].cast(FloatType()))
test = test.withColumn("user_id",test["user_id"].cast(IntegerType()))
test = test.withColumn("book_id",test["book_id"].cast(IntegerType()))
val = val.withColumn("rating",val["rating"].cast(FloatType()))
val = val.withColumn("user_id",val["user_id"].cast(IntegerType()))
val = val.withColumn("book_id",val["book_id"].cast(IntegerType()))

train_review_feature = spark.read.parquet('train_review25.parquet')
val_review_feature = spark.read.parquet('val_review25.parquet')
test_review_feature = spark.read.parquet('test_review25.parquet')
train_review_feature = train.join(train_review_feature,['user_id','book_id','rating']).select('user_id','book_id','rating','idf_features','is_read')
val_review_feature = val.join(val_review_feature,['user_id','book_id','rating']).select('user_id','book_id','rating','idf_features','is_read')
test_review_feature = test.join(test_review_feature,['user_id','book_id','rating']).select('user_id','book_id','rating','idf_features','is_read')

# is_read
train_read = train.filter(train.is_read != 0)
val_read = val.filter(val.is_read != 0)
test_read = test.filter(test.is_read != 0)
train_unread = train.filter(train.is_read == 0)
val_unread = val.filter(val.is_read == 0)
test_unread = test.filter(test.is_read == 0)
train_review_feature = train_review_feature.filter(train_review_feature.is_read != 0)
val_review_feature = val_review_feature.filter(val_review_feature.is_read != 0)
test_review_feature = test_review_feature.filter(test_review_feature.is_read != 0)

als=ALS(maxIter=5,regParam=0.0,rank=1,userCol='user_id',itemCol='book_id',ratingCol='rating',coldStartStrategy="drop",nonnegative=True)
model_unread = als.fit(train_unread)
predictions_unread = model_unread.transform(val_unread)

# linear regression
regParam = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]
netParam = [0.001,0.01,0.05,0.1,0.2]

window = Window.partitionBy(val_review_feature['user_id']).orderBy(val_review_feature['rating'].desc())
val_true_list = val_review_feature.select('*', rank().over(window).alias('true_row'))
##5%: reg=0.2, net=0.01; 25%: reg=0.1, net=0.01; 100%: reg=0.05, net=0.01;
best_model_rmse_lr,best_model_map_lr = tuning.lr_tune(train_review_feature, val_review_feature, val_true_list, regParam, netParam)

val_predictions = best_model_rmse_lr.transform(val_review_feature)
review_val_predictions = val_predictions.withColumn('prediction', when(val_predictions['prediction'] < 0, 0).otherwise(val_predictions['prediction']))

# true validation list
window = Window.partitionBy(val['user_id']).orderBy(val['rating'].desc())
val_true_list = val.select('*', rank().over(window).alias('true_row'))

# hyperparameter tuning
num_iters = [20]
reg_params = [0.01, 0.05, 0.1, 0.2, 0.5]
ranks = [10, 20]

best_model_rmse,best_model_map = tuning.tune_ALS_NLP_read(spark, train_read, val_read, val_true_list, num_iters, reg_params, ranks, review_val_predictions, predictions_unread)

# test performance
test_predictions = best_model_rmse_lr.transform(test_review_feature)
review_test_predictions = test_predictions.withColumn('prediction', when(test_predictions['prediction'] < 0, 0).otherwise(test_predictions['prediction']))
review_test_predictions = review_test_predictions.withColumnRenamed('prediction','review_prediction')

test_predictions = best_model_rmse.transform(test_read)
als_test_predictions = test_predictions.withColumnRenamed('prediction','als_prediction')

predictions_unread = model_unread.transform(test_unread)
predictions_unread = predictions_unread.withColumnRenamed('prediction','unread_prediction')

total_predictions = als_test_predictions.join(review_test_predictions,['user_id','book_id','rating'],'outer').join(predictions_unread,['user_id','book_id','rating'],'outer')
total_predictions = total_predictions.withColumn('total_prediction', when(total_predictions['review_prediction'].isNotNull(), total_predictions['review_prediction']).otherwise(total_predictions['als_prediction']))
total_predictions = total_predictions.withColumn('total_prediction', when(total_predictions['total_prediction'].isNull(),total_predictions['unread_prediction']).otherwise(total_predictions['total_prediction']))
window = Window.partitionBy(total_predictions['user_id']).orderBy(total_predictions['total_prediction'].desc())
top_predictions = total_predictions.select('*', rank().over(window).alias('row_num')).filter(col('row_num')<=500)

evaluator=RegressionEvaluator(metricName='rmse', labelCol='rating',predictionCol='total_prediction')
rmse_test = evaluator.evaluate(top_predictions)

window = Window.partitionBy(test['user_id']).orderBy(test['rating'].desc())
test_true_list = test.select('*', rank().over(window).alias('true_row'))
map_score = MAP.getMAP(top_predictions, test_true_list)
print('Test set RMSE = {}, Test set MAP = {}'.format(rmse_test, map_score))



