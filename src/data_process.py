import os
import sys
import copy
import time
import random
import math
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from sklearn import preprocessing
from pyspark.sql import SparkSession
import pyspark.sql.functions as pysparkk
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Data processing using pyspark") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def process_data(filename):
    '''
    Process the csv file to remove unwanted columns and convert date time to
    time in sec, year, day of month
    Also create a lable congestion status based on speed recorded in the data set
    speed >=20 Green
    speed 10-20 Yellow
    speed < 10 Red
    Return value: a data frame
    '''
    spark = init_spark()
    data = spark.read.load(filename, format='com.databricks.spark.csv', header='true',
                           inferSchema='true')
    data = data.filter(data.SPEED != -1)
    congestion_label = udf(checkCongestion_level, StringType())
    time_converstion = udf(convertTimeToSec, IntegerType())
    split_col = pysparkk.split(data['TIME'], ' ')
    split_col_date = pysparkk.split(split_col.getItem(0), '/')
    data = data.withColumn('TIME_IN_SEC', time_converstion(split_col.getItem(1), split_col.getItem(2)))
    data = data.withColumn("Congestion_Status", congestion_label(data.SPEED))
    data = data.withColumn('YEAR', split_col_date.getItem(2))
    data = data.withColumn('DAY_OF_MONTH', split_col_date.getItem(1))
    column_to_drop = ['Wards', 'Zip Codes', 'Community Areas', 'MESSAGE_COUNT', 'RECORD_ID', 'TIME', 'END_LOCATION',
                      'START_LOCATION', 'COMMENTS', 'STREET', 'BUS_COUNT', 'STREET_HEADING', 'FROM_STREET', 'TO_STREET',
                      'SEGMENT_ID', 'SPEED']
    data = data.drop(*column_to_drop)
    return data


def write_csv_data_single_file(data, filename):
    '''
    Write the procssed data frame as a single CSV file
    '''
    data.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("inferSchema",
                                                                                              "true").save(filename)


def checkCongestion_level(speed):
    '''
    Add a label congestion level based on the speed recorded
    '''
    status = ''
    if speed >= 20:
        status = 'Green'
    elif speed > 10 and speed < 20:
        status = 'Yello'
    else:
        status = 'Red'
    return status


def convertTimeToSec(time, am_pm):
    '''
    # Convert time to Sec
    '''
    if am_pm == 'PM':
        time = time.split(":")
        return (int(time[0]) + 12) * 3600 + int(time[1]) * 60 + int(time[2])
    else:
        time = time.split(":")
        if int(time[0]) == 12:
            return int(time[1]) * 60 + int(time[2])
        else:
            return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])


def execute(inputFile, outputFile):
    '''
    Execute the data file
    :param inputFile:
    :param outputFile:
    '''
    data = process_data(inputFile)
    write_csv_data_single_file(data, outputFile)

execute("dataset/chicago_traffic_2018.csv", "final_output")
