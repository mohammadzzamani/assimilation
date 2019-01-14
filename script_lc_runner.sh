#!/bin/ksh
set -v

#export PYSPARK_PYTHON=/opt/app/anaconda2/python27/bin/python;
export SPARK_HOME=/opt/mapr/spark/spark-2.0.1/
export PYSPARK_PYTHON=python2.7
export PYTHONPATH=.

#hadoop fs -get $2/files/* .

~/../../../opt/mapr/spark/spark-2.0.1/bin/spark-submit \
--master yarn \
--deploy-mode client \
--executor-memory 10g \
--driver-memory 10g \
--conf spark.yarn.executor.memoryOverhead=600 \
--jars ~/hadoopPERMA/jars/hadoop-lzo-0.4.21-SNAPSHOT.jar \
spark_lang_change_runner.py

#--driver-library-path /usr/hdp/current/hadoop-client/lib/native/Linux-amd64-64 \
#--num-executors 40 \
#--executor-cores 2 \
#--master yarn \
#--deploy-mode cluster \
#--executor-memory 10g \
#--driver-memory 15g \
#--conf spark.yarn.executor.memoryOverhead=1024 \
#--conf spark.sql.shuffle.partitions=1000 \
#--conf spark.io.compression.codec=lzf \
#--conf spark.network.timeout=800 \
#--conf spark.rpc.askTimeout=800 \
#--conf spark.locality.wait=10s \
#--conf spark.task.maxFailures=10000 \
#--conf spark.shuffle.manager=SORT \
#--conf spark.akka.frameSize=300 \
#--conf spark.yarn.max.executor.failures=1000 \
#--conf spark.shuffle.consolidateFiles=true \
#--conf spark.shuffle.service.enabled=true \
#--conf spark.default.parallelism=2000 \
#--conf spark.driver.maxResultSize=2048 \
#--py-files conf.zip,\
#spark_schema.zip,shared.zip,readers.zip,\
#goozie.zip,\
#descriptors.zip \
#--files /usr/hdp/current/spark-client/conf/hive-site.xml \
#--jars tez-api.jar,\
#datanucleus-rdbms.jar,datanucleus-core.jar,\
#datanucleus-api-jdo.jar,commons-csv-1.1.jar,spark-csv_2.10-1.5.0.jar \
#prod_rg_inventory.py run prod_rg_inventory rg_inventory $1