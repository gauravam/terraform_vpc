
from awsglue.context import GlueContext
from pyspark.context import SparkContext

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read from Glue Catalog using GlueContext
dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database="my_database",
    table_name="my_table",
    transformation_ctx="datasource"
)

# Convert to Spark DataFrame and limit to 5 rows
df = dynamic_frame.toDF().limit(5)

# Process data
df.show()
