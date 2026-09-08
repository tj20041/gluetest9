import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, concat, lit, coalesce
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

# Log any records with NULL cost_center before transformation so they are
# visible in the job log without needing to re-run the pipeline.
null_rows = payroll_df.filter(col("cost_center").isNull()).select("employee_id").collect()
if null_rows:
    logger.warning("NULL cost_center employee_ids: %s", [r.employee_id for r in null_rows])

logger.info("Applying secure hash identifier generation across workers...")

# Replace the Python UDF with native Catalyst expressions (concat + coalesce).
# This eliminates the JVM-to-Python serialisation boundary entirely and handles
# NULL column values safely without raising a TypeError.  coalesce() substitutes
# the literal 'UNKNOWN' whenever employee_code or cost_center is NULL, and
# casting the arithmetic result to StringType() mirrors the original str(points * 1.5).
hashed_df = payroll_df.withColumn(
    "audit_hash",
    concat(
        coalesce(col("employee_code"), lit("UNKNOWN")),
        lit ("::"),
        coalesce(col("cost_center"), lit("UNKNOWN")),
        lit ("::"),
        (col("bonus_points") * lit(1.5)).cast(StringType())
    )
)

# Trigger executor evaluation
hashed_df.collect()
job.commit()
