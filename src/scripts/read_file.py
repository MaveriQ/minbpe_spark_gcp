from pyspark.sql import SparkSession
from pyspark.rdd import RDD


def read_file(spark: SparkSession,
              file_uri: str) -> RDD:
    """
    :param spark: Spark Context
    :param file_uri: URI of the Input File
    :return: Spark RDD
    """
    return (spark
            .read
            .parquet(file_uri).rdd
            )
