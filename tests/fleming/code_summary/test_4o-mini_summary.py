import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from tests.conftest import spark_session

from src.fleming.code_summary.4o-mini_summary import OpenAIClient

@pytest.fixture
def input_spark_df(spark_session):
    data = [("repo1", "content1", "token1"), ("repo2", "content2", "token2")]
    return spark_session.createDataFrame(data, ["title", "concatenated_content", "total_token_count"])

@pytest.fixture
def openai_client(spark_session, input_spark_df):
    return OpenAIClient(spark_session, input_spark_df, "output_table", "prompt", "api_key", "endpoint")

@patch("requests.post")
def test_call_openai_success(mock_post, openai_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "summary"}}]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    openai_client.call_openai("title", "concatenated_content", "total_token_count")

    assert openai_client.results_df is not None
    assert openai_client.results_df.count() == 2

@patch("requests.post")
def test_call_openai_failure(mock_post, openai_client):
    mock_post.side_effect = requests.RequestException("API call failed")

    with pytest.raises(SystemExit):
        openai_client.call_openai("title", "concatenated_content", "total_token_count")

def test_display_results(openai_client):
    openai_client.results_df = openai_client.spark.createDataFrame(
        [("repo1", "prompt", "token1", "summary")],
        ["repo_name", "prompt", "repo_token_count", "virtual_readme"]
    )

    with patch("builtins.print") as mock_print:
        openai_client.display_results()
        mock_print.assert_called()

def test_display_results_no_results(openai_client):
    with pytest.raises(ValueError):
        openai_client.display_results()

def test_save_results(openai_client):
    openai_client.results_df = openai_client.spark.createDataFrame(
        [("repo1", "prompt", "token1", "summary")],
        ["repo_name", "prompt", "repo_token_count", "virtual_readme"]
    )

    with patch.object(openai_client.results_df, "write") as mock_write:
        openai_client.save_results()
        mock_write.mode.assert_called_with("overwrite")

def test_save_results_no_results(openai_client):
    with pytest.raises(ValueError):
        openai_client.save_results()