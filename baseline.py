#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
             .config("spark.shuffle.memoryFraction","0")
             .config("spark.memory.offHeap.enabled","true")
             .config("spark.memory.offHeap.size",memory)
             .getOrCreate())
sc.setCheckpointDir('checkpoint/')

# on cluster
"""
memory = "30g"
spark = (SparkSession.builder
             .appName('als')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.executor.cores','10')
             .config('spark.driver.memory', memory)
             .config('spark.driver.cores','10')
             .config('spark.executor.memoryOverhead', '4096')
             .config("spark.sql.broadcastTimeout", "36000")
             .config("spark.storage.memoryFraction","0")
             .config("spark.shuffle.memoryFraction","0")
             #.config("spark.memory.offHeap.enabled","true")
             .config("spark.memory.offHeap.size",memory)
             .getOrCreate())
sc.setCheckpointDir('hdfs:/user/liy31/checkpoint/')

"""

from pyspark.sql.types import DateType,IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
from pyspark.sql.functions import expr
from pyspark.mllib.evaluation import RankingMetrics,RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

train = spark.read.parquet("train25.parquet")
val = spark.read.parquet("val25.parquet")
val_true_list = spark.read.parquet("val25_true_list.parquet")

train = train.withColumn("rating",train["rating"].cast(FloatType()))
train = train.withColumn("user_id",train["user_id"].cast(IntegerType()))
train = train.withColumn("book_id",train["book_id"].cast(IntegerType()))

val = val.withColumn("rating",val["rating"].cast(FloatType()))
val = val.withColumn("user_id",val["user_id"].cast(IntegerType()))
val = val.withColumn("book_id",val["book_id"].cast(IntegerType()))


#num_iters = [5,10,15,20]
num_iters = [20]
reg_params = [0.001, 0.01, 0.05, 0.1, 0.2,0.4,0.6,0.8,1.0]
ranks = [8, 10, 12, 14, 16, 18, 20]

def tune_ALS_map(train_data, validation_data, val_true_list, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_iter1 = -1
    best_rank1 = -1
    best_regularization1 = 0
    best_model_rmse = None
    max_map = 0.0
    best_iter2 = -1
    best_rank2 = -1
    best_regularization2 = 0
    best_model_map = None
    for iteration in maxIter:
        for current_rank in ranks:
            for reg in regParams:
                # get ALS model
                #als = ALS().setMaxIter(iteration).setRank(rank).setRegParam(reg)
                als=ALS(maxIter=iteration,regParam=reg,rank=current_rank,
                        userCol='user_id',itemCol='book_id',ratingCol='rating',
                        coldStartStrategy="drop",nonnegative=True,checkpointInterval = 1)
                # train ALS model
                model = als.fit(train_data)
                # evaluate the model by computing the RMSE on the validation data
                predictions = model.transform(validation_data)
                window = Window.partitionBy(predictions['user_id']).orderBy(predictions['prediction'].desc())
                val_pred_order  = predictions.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 500)  
                evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
                rmse = evaluator.evaluate(val_pred_order)
            
                if rmse < min_error:
                    min_error = rmse
                    best_rank1 = current_rank
                    best_regularization1 = reg
                    best_iter1 = iteration
                    best_model_rmse = model
                # evaluate the model by computing the MAP on the validation data
                
                val_pred_list =val_pred_order.select('user_id','book_id').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
                val_RDD = val_pred_list.join(val_true_list, 'user_id').rdd.map(lambda row: (row[1], row[2]))
                rankingMetrics = RankingMetrics(val_RDD)
                current_map = rankingMetrics.meanAveragePrecision
                #pk = rankingMetrics.precisionAt(500)
                
                    
                if current_map > max_map:
                    max_map = current_map
                    best_rank2 = current_rank
                    best_regularization2 = reg
                    best_iter2 = iteration
                    best_model_map = model
                    
                print('{} latent factors and regularization = {} with maxIter {}: '
                  'validation RMSE is {}' 'validation MAP is {}' .format(current_rank, reg, iteration, rmse, current_map))

                with open('train25_eval.csv', 'ab') as f:
                    np.savetxt(f, [np.array([iteration, current_rank, reg, rmse, current_map])],delimiter=",")
                    
    print('\nThe best model select by RMSE has {} latent factors and '
          'regularization = {}''with maxIter = {}'.format(best_rank1, best_regularization1, best_iter1))
    print('\nThe best model select by MAP has {} latent factors and '
          'regularization = {}''with maxIter = {}'.format(best_rank2, best_regularization2, best_iter2))
    
    
    return best_model_rmse,best_model_map

best_model_rmse,best_model_map = tune_ALS_map(train,val,val_true_list, num_iters, reg_params, ranks)
#best_model_rmse.save("best_model01_rmse")
#best_model_map.save("best_model01_map")
