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

# Data-quality guard: surface null cost_center rows via CloudWatch logs
# instead of letting them silently crash the UDF/task 4x and abort the job.
null_cost_center_count = payroll_df.filter(col("cost_center").isNull()).count()
if null_cost_center_count > 0:
    logger.warning(
        "Found %d payroll record(s) with NULL cost_center. "
        "These will be coalesced to 'UNKNOWN' before hashing.",
        null_cost_center_count
    )

# Defensive upstream fillna so the UDF never receives None for cost_center
# or employee_code, even if the null-guard inside the UDF is bypassed.
payroll_df = payroll_df.fillna({
    "employee_code": "UNKNOWN",
    "cost_center": "UNKNOWN"
})


def generate_payroll_hash(emp_code, cost_center, points):
    """Build a compliance audit hash for a payroll record.

    Null-safe: coalesces None values for emp_code/cost_center to
    'UNKNOWN' and None points to 0 before concatenation, and wraps the
    computation in a try/except so a single malformed record returns a
    sentinel value ('HASH_ERROR') instead of raising a TypeError that
    would kill the Spark task and abort the whole Glue job.
    """
    try:
        safe_emp_code = emp_code if emp_code is not None else "UNKNOWN"
        safe_cost_center = cost_center if cost_center is not None else "UNKNOWN"
        safe_points = points if points is not None else 0
        return safe_emp_code + "::" + safe_cost_center + "::" + str(safe_points * 1.5)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to generate payroll hash for record: %s", exc)
        return "HASH_ERROR"


payroll_hash_udf = udf(generate_payroll_hash, StringType())

logger.info("Applying secure hash identifier generation across workers...")

hashed_df = payroll_df.withColumn(
    "audit_hash",
    payroll_hash_udf(col("employee_code"), col("cost_center"), col("bonus_points"))
)

# Trigger executor evaluation
hashed_df.collect()
job.commit()
