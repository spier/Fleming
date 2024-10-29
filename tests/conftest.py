import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """
    Fixture for creating a Spark session for testing.
    This Spark session will be shared across all tests in the session.
    """
    return SparkSession.builder.appName("-test").getOrCreate()
