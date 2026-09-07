import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, StructType, StructField, IntegerType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payroll_enrichment")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Loading payroll transaction audit batch...")

payroll_schema = StructType([
    StructField("employee_id", IntegerType(), False),
    StructField("employee_code", StringType(), True),
    StructField("cost_center", StringType(), True),
    StructField("bonus_points", IntegerType(), True)
])

payroll_data = [
    (1001, "EMP-A", "CC-Finance", 50),
    (1002, "EMP-B", None, 25),            # Contains None in cost center
    (1003, "EMP-C", "CC-Operations", 10)
]

payroll_df = spark.createDataFrame(payroll_data, payroll_schema)

def generate_payroll_hash(emp_code, cost_center, points):
    # FAILS HERE: Fails on record 1002 where cost_center is None: TypeError: can only concatenate str to str
    return emp_code + "::" + cost_center + "::" + str(points * 1.5)

payroll_hash_udf = udf(generate_payroll_hash, StringType())

logger.info("Applying secure hash identifier generation across workers...")

hashed_df = payroll_df.withColumn(
    "audit_hash",
    payroll_hash_udf(col("employee_code"), col("cost_center"), col("bonus_points"))
)

# Trigger executor evaluation
hashed_df.collect()
job.commit()
