from pyspark.sql.types import StructType, StructField, StringType
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import DataFrame
import constants as c
import os

topic_table = 'feat.2000fb.msg_1001_nrt.userid_week'
ngram_table = 'feat.1gram.msg_1001_nrt_small.userid_week'
accounts_table = 'twitter_accounts_1001_uniques.csv'
network_table = 'twitter_network_1001.csv'

def build_spark_context_cluster(app_name):
    global sc
    conf = SparkConf()
    conf.setAppName(app_name)
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sc = SQLContext(sc)
    return sc

# def read_data_frame(file_name, schema, has_header='false', delimiter='\001', compress=None):
def read_data_frame(file_name, schema=None, has_header='false', delimiter=',', compress=None):
    """
    :param schema:
    :type schema: pyspark.sql.types.StructType
    :param file_name: full HDFS path of CSV file name
    :param has_header: only applicable when schema is passed
    :type has_header: str ('false' or 'true')
    :type file_name: str
    :return:
    :rtype: DataFrame
    """
    global sc
    if schema is None:
        if compress:
            assert compress in [c.GZIP, c.SNAPPY]
            df = sc.read \
                .format('com.databricks.spark.csv') \
                .options(header='true', inferschema='true', delimiter=delimiter, codec=compress) \
                .load(file_name)
        else:
            df = sc.read \
                .format('com.databricks.spark.csv') \
                .options(header='true', inferschema='true', delimiter=delimiter) \
                .load(file_name)
    else:
        if compress:
            assert compress in [c.GZIP, c.SNAPPY]
            df = sc.read \
                .format('com.databricks.spark.csv') \
                .options(header=has_header, inferschema='false', delimiter=delimiter, codec=compress) \
                .schema(schema) \
                .load(file_name)
        else:
            df = sc.read \
                .format('com.databricks.spark.csv') \
                .options(header=has_header, inferschema='false', delimiter=delimiter) \
                .schema(schema) \
                .load(file_name)

        df.printSchema()
        print ('df.count: ', df.count())
        print ('df.columns: ', df.columns)
        df = df.repartition(100)
    return df

def write_data_frame(df, output_path, delimiter=',', repartition=None, coalesce=None, compress=None):
    global sc
    if repartition is not None:
        df = df.repartition(repartition)
    if coalesce is not None:
        df = df.coalesce(coalesce)
    df.cache()

    if compress:
        assert compress in [c.GZIP, c.SNAPPY]
        df.write \
            .options(header='true', delimiter=delimiter) \
            .option("codec", compress) \
            .format('com.databricks.spark.csv') \
            .mode("overwrite") \
            .save(output_path)
    else:
        df.write \
            .options(header='true', delimiter=delimiter) \
            .format('com.databricks.spark.csv') \
            .mode("overwrite") \
            .save(output_path)

def run_command(cmd):
    """Return (status, output) of executing cmd in a shell."""
    pipe = os.popen('{ ' + cmd + '; } 2>&1', 'r')
    # text = pipe.read()
    # sts = pipe.close()
    # if sts is None: sts = 0
    # if text[-1:] == '\n': text = text[:-1]
    # return sts, text



if __name__ == "__main__":
    global sc
    sc = build_spark_context_cluster( 'running my bundle' )
    ROOT_DIR = '/user/mzamani/'


    #### read topics_df, accounts_df, and network_df:
    topics_df = read_data_frame(file_name=ROOT_DIR + topic_table, schema=StructType([StructField(field, StringType(), True) for field in ['group_id', 'feat', 'value', 'group_norm']]), compress = 'gzip' )
    topics_df.cache()
    print ('topics_df: ')
    topics_df.show()

    accounts_df = read_data_frame(file_name=ROOT_DIR + accounts_table, schema=None, compress = 'gzip' )
    accounts_df.cache()
    print ('accounts_df: ')
    accounts_df.show()

    network_df = read_data_frame(file_name=ROOT_DIR + network_table, schema=None, compress = 'gzip' )
    network_df.cache()
    print ('network_df: ')
    network_df.show()
    ####



    output_dir = ROOT_DIR + 'test_dir/'
    full_output_path = output_dir + 'test1'
    run_command('hadoop fs -mkdir -p %s' % output_dir)
    write_data_frame(accounts_df, full_output_path, repartition=100)

