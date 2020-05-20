#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql import Window
from pyspark.sql import functions as F

def getMAP(top_predictions, truth):
    true = truth.select('user_id','book_id','true_row') 
    w = Window.partitionBy('user_id').orderBy('true_row')
    true = true.withColumn('true',F.collect_list('book_id').over(w)).groupBy('user_id').agg(F.max('true').alias('true'))

    pred = top_predictions.select('user_id','book_id','row_num')
    w = Window.partitionBy('user_id').orderBy('row_num')
    pred = pred.withColumn('pred',F.collect_list('book_id').over(w)).groupBy('user_id').agg(F.max('pred').alias('pred'))

    pred_true=pred.join(true,'user_id').select('pred','true').rdd

    metrics = RankingMetrics(pred_true)
    score = metrics.meanAveragePrecision
    return score
