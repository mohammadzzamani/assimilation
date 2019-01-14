from pyspark.sql.types import StructType, StructField, StringType
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import DataFrame
from pyspark.sql.functions import first, split, col
# import constants as c
import os


GZIP = 'gzip'
SNAPPY = 'snappy'
ROOT_DIR = '/user/mzamani/'
topics_table = 'feat.2000fb.msg_1001_nrt_small.userid_week'
ngrams_table = 'feat.1gram.msg_1001_nrt_small.userid_week'
accounts_table = 'twitter_accounts_1001_uniques'
network_table = 'twitter_network_1001'
core_users_weeks_table = 'core_users_weeks'

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
            assert compress in [GZIP, SNAPPY]
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
            assert compress in [GZIP, SNAPPY]
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

def write_data_frame(df, output_path, delimiter=',', repartition=None, coalesce=None, compress=None, show_schema=True):
    global sc
    if repartition is not None:
        df = df.repartition(repartition)
    if coalesce is not None:
        df = df.coalesce(coalesce)
    df.cache()

    if show_schema:
        print ('df.schema: ')
        df.printSchema()

    if compress:
        assert compress in [GZIP, SNAPPY]
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
    print ('run_command: ', cmd)
    """Return (status, output) of executing cmd in a shell."""
    pipe = os.popen('{ ' + cmd + '; } 2>&1', 'r')
    # text = pipe.read()
    # sts = pipe.close()
    # if sts is None: sts = 0
    # if text[-1:] == '\n': text = text[:-1]
    # return sts, text


def read_input():

    print ('topics_df: ')
    topics_schema = StructType([StructField(field, StringType(), True) for field in
                                ['group_id', 'feat', 'value', 'group_norm']])
    topics_df = read_data_frame(file_name=ROOT_DIR + topics_table, schema=topics_schema, compress = 'gzip' )
    topics_df.cache()

    print ('ngrams_df: ')
    ngrams_df = read_data_frame(file_name=ROOT_DIR + ngrams_table, schema=topics_schema, compress = 'gzip' )
    ngrams_df.cache()

    print ('accounts_df: ')
    accounts_schema = StructType([StructField(field, StringType(), True) for field in
                                  ['counter' ,'id' ,'username' ,'last_message_id' ,'followers_count' ,'friends_count'
                                      ,'created_at' ,'favourites_count' ,'statuses_count' ,'level'
                                      ,'recorded_friends_count' ,'recorded_statuses_count']])
    accounts_df = read_data_frame(file_name=ROOT_DIR + accounts_table, schema=accounts_schema, compress = 'gzip' )
    accounts_df.cache()
    accounts_df.show()

    print ('core_users_weeks_df: ')
    core_users_weeks_schema = StructType([StructField(field, StringType(), True) for field in
                                  ["user_id","_min","_tercile1","_mean","_tercile2","_max","week_min",
                                   "week_tercile1","week_mean","week_tercile2","week_max"]])
    core_users_weeks_df = read_data_frame(file_name=ROOT_DIR + core_users_weeks_table, schema=core_users_weeks_schema, compress = 'gzip' )
    core_users_weeks_df.cache()
    core_users_weeks_df.show()

    print ('network_df: ')
    network_schema = StructType([StructField(field, StringType(), True) for field in
                                  ["counter","source_id","dest_id","likes","mentions","retweets","friends_since"] ])
    network_df = read_data_frame(file_name=ROOT_DIR + network_table, schema=network_schema, compress = 'gzip' )
    network_df.cache()
    network_df.show()

    return topics_df, ngrams_df, accounts_df, core_users_weeks_df, network_df



if __name__ == "__main__":
    global sc
    sc = build_spark_context_cluster( 'running my bundle' )


    ######## read topics_df, ngrams_df, accounts_df, core_users_weeks_df, network_df
    topics_df, ngrams_df, accounts_df, core_users_weeks_df, network_df = read_input()



    ######## select two and three hops users
    three_hops_users = accounts_df.where(accounts_df.level == 3)
    three_hops_users_list  = [int(row.id) for row in three_hops_users.collect()]
    two_hops_users = accounts_df.where(accounts_df.level == 2)
    two_hops_users_list  = [int(row.id) for row in two_hops_users.collect()]
    print ('three_hops_users: ')
    print (three_hops_users.show())
    print (three_hops_users.count())



    ######## building  all_users_indices dictionary: {userid: int index}
    all_users_indices = { two_hops_users_list[i] : i for i in range(len(two_hops_users_list)) }
    print ('(len(self.all_users_indices): ', len(all_users_indices ))
    for uid in  three_hops_users_list:
            if uid not in two_hops_users_list:
                    all_users_indices[uid] = len(all_users_indices)
    print ('(len(self.all_users_indices): ', len(all_users_indices ))



    ######## split userid_week into two columns in topics_df
    topics_df = topics_df.select(['group_id', 'feat', 'group_norm'])
    topics_pivoted_df = topics_df.groupby(topics_df.group_id).pivot("feat").agg(first("group_norm"))
    split_groupid_col = split(topics_pivoted_df['group_id'], '_')
    topics_pivoted_df = topics_pivoted_df.withColumn('id_only', split_groupid_col.getItem(0)).withColumn('week_only', split_groupid_col.getItem(1))
    print ('topics_pivoted_df:')
    topics_pivoted_df.show(n=1)



    #### write topics_pivoted_df dataframe to hadoop storage
    output_dir = ROOT_DIR + 'data_dir/'
    full_output_path = output_dir + 'topics_pivoted_df'
    run_command('hadoop fs -mkdir -p %s' % output_dir)
    run_command('hadoop fs -rm -r -skipTrash %s' % full_output_path)
    print ('topics_pivoted_df.count: ', topics_pivoted_df.count())
    topics_pivoted_df = topics_pivoted_df.sample(False,0.1, 42)
    print ('topics_pivoted_df.count after sampling: ', topics_pivoted_df.count())
    write_data_frame(topics_pivoted_df, full_output_path, repartition=100, compress=None, show_schema=False)


    ######## store topics_pivoted_df schema in a file
    topics_columns_file = open('~/assimilation/data_dir/topics_columns.txt')
    topics_columns_file.write(topics_pivoted_df.columns)


    topics_pivoted_df = topics_pivoted_df.alias('a').join(core_users_weeks_df.alias('b'), col('a.id_only') == col('b.user_id') ).\
        select([col('a.'+c) for c in topics_pivoted_df.columns] + [col('b.week_mean')])
    print ('topics_pivoted_df.columns: ', topics_pivoted_df.columns)

    topics_first_df = topics_pivoted_df.where(col('week_only') < col('week_mean'))
    topics_last_df = topics_pivoted_df.where(col('week_only') >= col('week_mean'))





    #### write a dataframe to hadoop storage
    output_dir = ROOT_DIR + 'test_dir/'
    full_output_path = output_dir + 'test2'
    run_command('hadoop fs -mkdir -p %s' % output_dir)
    write_data_frame(accounts_df, full_output_path, repartition=100, compress=None)

