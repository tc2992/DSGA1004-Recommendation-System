#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this script is for evaluating performance of baseline model and read model on test set
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
# on local

import math
import psutil
memory = f"{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g"
spark = (SparkSession.builder
             .appName('als')
             .master('local[*]')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .config('spark.executor.memoryOverhead', '4096')
             .config("spark.sql.broadcastTimeout", "36000")
             .config("spark.storage.memoryFraction","0")
             .config("spark.memory.offHeap.enabled","true")
             .config("spark.memory.offHeap.size",memory)
             .getOrCreate())
sc.setCheckpointDir('checkpoint/')

# on cluster
"""
memory = "15g"
spark = (SparkSession.builder
             .appName('als')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.executor.cores',4)
             .config('spark.driver.memory', memory)
             .config('spark.executor.memoryOverhead', '4096')
             .config("spark.sql.broadcastTimeout", "36000")
             .config("spark.storage.memoryFraction","0")
             .config("spark.memory.offHeap.enabled","true")
             .config("spark.memory.offHeap.size",memory)
             .getOrCreate())

sc.setCheckpointDir('hdfs:/user/liy31/checkpoint/')
"""


from pyspark.sql.types import DateType,IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col, expr, when
from pyspark.mllib.evaluation import RankingMetrics,RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator


train = spark.read.parquet("train100.parquet")
test = spark.read.parquet("test100.parquet")
test_true_list = spark.read.parquet("test_true_list/test100_true_list.parquet")

train = train.withColumn("rating",train["rating"].cast(FloatType()))
train = train.withColumn("user_id",train["user_id"].cast(IntegerType()))
train = train.withColumn("book_id",train["book_id"].cast(IntegerType()))

test = test.withColumn("rating",test["rating"].cast(FloatType()))
test = test.withColumn("user_id",test["user_id"].cast(IntegerType()))
test = test.withColumn("book_id",test["book_id"].cast(IntegerType()))


## evaluate baseline model
# best param for baseline model
iteration = 20
reg = 0.1
current_rank = 20

als=ALS(maxIter=iteration,regParam=reg,rank=current_rank,userCol='user_id',itemCol='book_id',ratingCol='rating',coldStartStrategy="drop",nonnegative=True)
# train ALS model
model = als.fit(train)
# evaluate the model by computing the RMSE on the validation data
predictions = model.transform(test)
window = Window.partitionBy(predictions['user_id']).orderBy(predictions['prediction'].desc())
test_pred_order  = predictions.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 500)
          
evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
rmse = evaluator.evaluate(test_pred_order)
            
# evaluate the model by computing the MAP on the validation data
test_pred_list = test_pred_order.select('user_id','book_id').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
test_RDD = test_pred_list.join(test_true_list, 'user_id').rdd.map(lambda row: (row[1], row[2]))
rankingMetrics = RankingMetrics(test_RDD)
current_map = rankingMetrics.meanAveragePrecision
              
print('\nThe best baseline model select by RMSE = {} has {} latent factors and '
          'regularization = {}  with maxIter = {} MAP = {}'.format(rmse,current_rank, reg, iteration,current_map))


"""
# evaluate read model

train_new = train.withColumn('rating',when(train.is_read == 0,float('nan')).otherwise(train.rating))
train_read = train_new.na.drop()
train_unread = train.subtract(train_read)

test_new = test.withColumn('rating',when(test.is_read == 0,float('nan')).otherwise(test.rating))
test_read = test_new.na.drop()
test_unread = test.subtract(test_read)

# best param for read model
iteration = 20
reg = 0.2
current_rank = 20

#built unread model first 
als=ALS(maxIter=5,regParam=0.0,rank=10,userCol='user_id',itemCol='book_id',ratingCol='rating',coldStartStrategy="drop",nonnegative=True)
model_unread = als.fit(train_unread)
predictions_unread = model_unread.transform(test_unread)

als=ALS(maxIter=iteration,regParam=reg,rank=current_rank,userCol='user_id',itemCol='book_id',ratingCol='rating',coldStartStrategy="drop",nonnegative=True)
# train ALS model
model_read = als.fit(train_read)
# evaluate the model by computing the RMSE on the validation data
predictions_read = model_read.transform(test_read)
        
predictions_all = predictions_read.union(predictions_unread)
window = Window.partitionBy(predictions_all['user_id']).orderBy(predictions_all['prediction'].desc())
test_pred_order  = predictions_all.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 500)  
       
evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
rmse = evaluator.evaluate(test_pred_order)
            
# evaluate the model by computing the MAP on the validation data
test_pred_list = test_pred_order.select('user_id','book_id').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
test_RDD = test_pred_list.join(test_true_list, 'user_id').rdd.map(lambda row: (row[1], row[2]))
rankingMetrics = RankingMetrics(test_RDD)
current_map = rankingMetrics.meanAveragePrecision
              
print('\nThe best read model select by RMSE = {} has {} latent factors and '
          'regularization = {}  with maxIter = {} MAP = {}'.format(rmse,current_rank, reg, iteration,current_map))

"""
