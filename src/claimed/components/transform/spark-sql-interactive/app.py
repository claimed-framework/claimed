from pyspark.sql import SparkSession

#stocator_jar = '/home/romeokienzler/gitco/claimed/component-library/transform/spark-sql-interactive/stocator-1.1.5-jar-with-dependencies.jar'
stocator_jar = '/opt/app-root/src/stocator/target/stocator-1.1.5-jar-with-dependencies.jar'
spark_builder = (
    SparkSession
        .builder
        .appName('test_app'))

spark_builder.config('spark.driver.extraClassPath', stocator_jar)
spark_builder.config('spark.executor.extraClassPath', stocator_jar)
spark_builder.config('fs.cos.myCos.access.key', 'afcfd8cda8dd4b17856577ec654c42e7')
spark_builder.config('fs.cos.myCos.secret.key', '')
spark_builder.config('fs.cos.myCos.endpoint', 's3.eu-de.cloud-object-storage.appdomain.cloud')
spark_builder.config("spark.hadoop.fs.stocator.scheme.list", "cos")
spark_builder.config("spark.hadoop.fs.cos.impl", "com.ibm.stocator.fs.ObjectStoreFileSystem")
spark_builder.config("fs.stocator.cos.impl", "com.ibm.stocator.fs.cos.COSAPIClient")
spark_builder.config("fs.stocator.cos.scheme", "cos")

spark = spark_builder.getOrCreate()
df = spark.read.csv("cos://claimed-spark-interactive.myCos/chat-en_us.csv")
df.createOrReplaceTempView('df')
df = spark.sql('select * from df')
#df.write.csv('cos://claimed-spark-interactive.myCos/chat-en_us3.csv')
df.writeTo("ibdb").create()

print("count:"+str(df.count()))
