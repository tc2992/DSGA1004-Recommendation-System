#!/usr/bin/env python
# -*- coding: utf-8 -*-

test = spark.read.csv('goodreads_interactions.csv',header = 'true')
test.createOrReplaceTempView('test')
#test.describe()
DataFrame[summary: string, user_id: string, book_id: string, is_read: string, rating: string, is_reviewed: string]

#test.stat.crosstab("is_read","rating").show()

#test.count()
#228648343
users = test.select('user_id').distinct()
users.count()
#876145

# sample 1% users
user01 =users.sample(False,0.01,seed = 2020)
user01.count()
#8849
# sample 5% users
user05 =users.sample(False,0.05,seed = 2020)
user05.count()
# sample 25% users
user25 =users.sample(False,0.05,seed = 2020)
user25.count()

dat = test.join(user01,['user_id'])
#dat = test.join(user05,['user_id']) 
#dat = test.join(user25,['user_id'])
#dat = test
dat.select('user_id').distinct().count()
#8849
dat.count()
#2256010
dat.write.parquet('sample01.parquet')

###### read x% sample of the whole data
dat = spark.read.parquet('sample01.parquet')

#describe data
ratings = dat.rdd
numRatings = ratings.count()
numUsers = ratings.map(lambda r: r[0]).distinct().count()
numBooks = ratings.map(lambda r: r[1]).distinct().count()

print("Got %d ratings from %d users on %d books." %(numRatings, numUsers, numBooks))
#Got 2256010 ratings from 8849 users on 516416 books.

## train,test,val split (the numbers in comment are for 1% data)
users = dat.select('user_id').distinct()
train_user,val_user,test_user = users.randomSplit([.6,.2,.2],seed = 2020)
train = dat.join(train_user,['user_id'])
train.count()
#1294419

val = dat.join(val_user,['user_id'])
val.count()
#489053
test = dat.join(test_user,['user_id'])
test.count()
#472538
from pyspark.sql.functions import lit
fractions = val.select('user_id').distinct().withColumn('fraction', lit(0.5)).rdd.collectAsMap()
val_add = val.sampleBy('user_id',fractions,seed = 2020)
val_add.count()
#244276
val_left = val.subtract(val_add)
val_left.count()
#244777

val_add.select('user_id').distinct().count()
#1792 
val_left.select('user_id').distinct().count()
#1788
#val_left.orderBy("user_id").groupBy("user_id").count().show(5)
df3 = val_left.join(val_add,'user_id','inner')
df3.select('user_id').distinct().count()
#1751, 37 users has no training data

### select only those userse appeared in training data
val_add_user = val_add.select('user_id').distinct()
val_new = val_left.join(val_add_user,['user_id'])
val_new.count()
#244704
val_new.select(['user_id']).distinct().count()
#1751
train_add_val = train.union(val_add)

fractions = test.select('user_id').distinct().withColumn('fraction', lit(0.5)).rdd.collectAsMap()
test_add = test.sampleBy('user_id',fractions,seed = 2020)
test_add.count()
#236144
test_left = test.subtract(test_add)
test_left.count()
#236394

test_add_user = test_add.select('user_id').distinct()
test_new = test_left.join(test_add_user,['user_id'])
test_new.count()
#236339
train_add_test = train_add_val.union(test_add)
train_add_test.select(['user_id']).distinct().count()
#8774
train.select(['user_id']).distinct().count()
#5298
val_new.select(['user_id']).distinct().count()
#1751
test_new.select(['user_id']).distinct().count()
#1636
## there are some users in val_add and test_add but not in val_new and test_new, therefore, 8774> 5298+1751+1636=8685

## example of writing x% data
train_add_test.write.parquet("train01.parquet")
val_new.write.parquet("val01.parquet")
test_new.write.parquet("test01.parquet")

# create the true rank list (example of 1% data)
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col

window = Window.partitionBy(val['user_id']).orderBy(val['rating'].desc())
val_true_order  = val.select('*', rank().over(window).alias('rank'))
val_true_list = val_true_order.select('user_id','book_id').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
val_true_list.write.parquet("val01_true_list.parquet")

window = Window.partitionBy(test['user_id']).orderBy(test['rating'].desc())
test_true_order  = test.select('*', rank().over(window).alias('rank'))
test_true_list = test_true_order.select('user_id','book_id').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
test_true_list.write.parquet("test01_true_list.parquet")


