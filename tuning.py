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

def lr_tune(train_data, validation_data,val_true_list,regParam,netParam):
    # initial
    min_error = float('inf')
    best_reg1 = None
    best_net1 = None
    best_model_rmse = None
    max_map = 0.0
    best_reg2 = None
    best_net2 = None
    best_model_map = None

    for reg in regParam:
        for net in netParam:
            lr = LinearRegression(featuresCol='idf_features',labelCol='rating',regParam=reg, elasticNetParam=net,maxIter=200)
            model = lr.fit(train_data)
            predictions = model.transform(validation_data)
            predictions = predictions.withColumn('prediction', when(predictions['prediction'] < 0, 0).otherwise(predictions['prediction']))

            # rmse
            evaluator=RegressionEvaluator(metricName='rmse', labelCol='rating',predictionCol='prediction')
            rmse = evaluator.evaluate(predictions)
            if rmse < min_error:
                min_error = rmse
                best_reg1 = reg
                best_net1 = net
                best_model_rmse = model

            # MAP top 25
            window = Window.partitionBy(predictions['user_id']).orderBy(predictions['prediction'].desc())
            top_predictions = predictions.select('*', rank().over(window).alias('row_num')).filter(col('row_num') <= 25)
            current_map = MAP.getMAP(top_predictions, val_true_list)
            if current_map > max_map:
                max_map = current_map
                best_reg2 = reg
                best_net2 = net
                best_model_map = model

            print('regParam = {} with elasticNetParam = {}: validation RMSE is {} validation MAP is {}'.format(reg, net, rmse, current_map))
    
    print('The best model select by RMSE has regParam = {} with elasticNetParam = {}: RMSE = {}'.format(best_reg1, best_net1, min_error))
    print('The best model select by MAP has regParam = {} with elasticNetParam = {}: MAP = {}'.format(best_reg2, best_net2, max_map))
    
    return best_model_rmse,best_model_map


def tune_ALS_NLP(spark, train_data, validation_data, val_true_list, maxIter, regParams, ranks, review_val_predictions):
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
                als=ALS(maxIter=iteration,regParam=reg,rank=current_rank, \
                        userCol='user_id',itemCol='book_id',ratingCol='rating', \
                        coldStartStrategy="drop",nonnegative=True)
                als_model = als.fit(train_data)
                predictions = als_model.transform(validation_data)
                
                review_predictions = review_val_predictions.withColumnRenamed('prediction','review_prediction')
                als_predictions = predictions.withColumnRenamed('prediction','als_prediction')
                total_predictions = als_predictions.join(review_predictions,['user_id','book_id','rating'],'outer')
                total_predictions = total_predictions.withColumn('total_prediction', \
                                                                 when(total_predictions['review_prediction'].isNotNull(), \
                                                                      total_predictions['review_prediction']) \
                                                                 .otherwise(total_predictions['als_prediction']))
                              
                window = Window.partitionBy(total_predictions['user_id']).orderBy(total_predictions['total_prediction'].desc())
                top_predictions = total_predictions.select('*', rank().over(window).alias('row_num')).filter(col('row_num') <= 500)

                # rmse
                evaluator=RegressionEvaluator(metricName='rmse', labelCol='rating',predictionCol='total_prediction')
                rmse = evaluator.evaluate(top_predictions)
                if rmse < min_error:
                    min_error = rmse
                    best_rank1 = current_rank
                    best_regularization1 = reg
                    best_iter1 = iteration
                    best_model_rmse = als_model

                # MAP
                current_map = MAP.getMAP(top_predictions, val_true_list)
                if current_map > max_map:
                    max_map = current_map
                    best_rank2 = current_rank
                    best_regularization2 = reg
                    best_iter2 = iteration
                    best_model_map = als_model

                print('{} latent factors and regularization = {} with maxIter {}: '
                  'validation RMSE is {}' 'validation MAP is {}' .format(current_rank, reg, iteration, rmse, current_map))
              
                with open('train05_review_eval.csv', 'ab') as f:
                    np.savetxt(f, [np.array([iteration, current_rank, reg, rmse, current_map])],delimiter=",")

    print('\nThe best model select by RMSE has {} latent factors and '
          'regularization = {}'' with maxIter = {}: RMSE = {}'.format(best_rank1, best_regularization1, best_iter1, min_error))
    print('\nThe best model select by MAP has {} latent factors and '
          'regularization = {}'' with maxIter = {}: MAP = {}'.format(best_rank2, best_regularization2, best_iter2, max_map))

    return best_model_rmse,best_model_map


def tune_ALS_NLP_read(spark, train_data, validation_data, val_true_list, maxIter, regParams, ranks, review_val_predictions, predictions_unread):
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
                als=ALS(maxIter=iteration,regParam=reg,rank=current_rank, \
                        userCol='user_id',itemCol='book_id',ratingCol='rating', \
                        coldStartStrategy="drop",nonnegative=True)
                als_model = als.fit(train_data)
                predictions = als_model.transform(validation_data)
                
                review_predictions = review_val_predictions.withColumnRenamed('prediction','review_prediction')
                als_predictions = predictions.withColumnRenamed('prediction','als_prediction')
                predictions_unread = predictions_unread.withColumnRenamed('prediction','unread_prediction')
                total_predictions = als_predictions.join(review_predictions,['user_id','book_id','rating'],'outer').join(predictions_unread,['user_id','book_id','rating'],'outer')
                total_predictions = total_predictions.withColumn('total_prediction', \
                                                                 when(total_predictions['review_prediction'].isNotNull(), \
                                                                      total_predictions['review_prediction']) \
                                                                 .otherwise(total_predictions['als_prediction']))
                total_predictions = total_predictions.withColumn('total_prediction', \
                                                                 when(total_predictions['total_prediction'].isNull(), \
                                                                      total_predictions['unread_prediction']) \
                                                                 .otherwise(total_predictions['total_prediction']))
                            
                window = Window.partitionBy(total_predictions['user_id']).orderBy(total_predictions['total_prediction'].desc())
                top_predictions = total_predictions.select('*', rank().over(window).alias('row_num')).filter(col('row_num') <= 500)

                # rmse
                evaluator=RegressionEvaluator(metricName='rmse', labelCol='rating',predictionCol='total_prediction')
                rmse = evaluator.evaluate(top_predictions)
                if rmse < min_error:
                    min_error = rmse
                    best_rank1 = current_rank
                    best_regularization1 = reg
                    best_iter1 = iteration
                    best_model_rmse = als_model

                # MAP
                current_map = MAP.getMAP(top_predictions, val_true_list)
                if current_map > max_map:
                    max_map = current_map
                    best_rank2 = current_rank
                    best_regularization2 = reg
                    best_iter2 = iteration
                    best_model_map = als_model

                print('{} latent factors and regularization = {} with maxIter {}: '
                  'validation RMSE is {}' 'validation MAP is {}' .format(current_rank, reg, iteration, rmse, current_map))
              
                with open('train05_combined_eval.csv', 'ab') as f:
                    np.savetxt(f, [np.array([iteration, current_rank, reg, rmse, current_map])],delimiter=",")

    print('\nThe best model select by RMSE has {} latent factors and '
          'regularization = {}'' with maxIter = {}: RMSE = {}'.format(best_rank1, best_regularization1, best_iter1, min_error))
    print('\nThe best model select by MAP has {} latent factors and '
          'regularization = {}'' with maxIter = {}: MAP = {}'.format(best_rank2, best_regularization2, best_iter2, max_map))

    return best_model_rmse,best_model_map
