# Copyright 2024 Fleming
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Databricks notebook source
#!pip install requests
#!pip install openai



#dbutils.library.restartPython()



import os
import requests
import base64
import json
import re



# input source delta table and output table in workflow parameters. Table name should have number indicating token sampling rate 

delta_table = dbutils.widgets.get("delta_table")
output_table_name = dbutils.widgets.get("output_table")
output_table_path = f"hive_metastore.dev_innersource.{output_table_name}"


def Call_OpenAI():
  API_KEY = dbutils.secrets.get(scope="Azure-OpenAI", key="ybdevgpt3-key-1")
  headers = {
      "Content-Type": "application/json",
      "api-key": API_KEY,
  }
  ENDPOINT = dbutils.secrets.get(scope="Azure-OpenAI", key="4o-mini-endpoint")
  # create list for results to be appended to
  results = []

  repo_contents_df = spark.read.table(delta_table)
  #repo_contents_df = repo_contents_df.limit(10)
  for row in repo_contents_df.collect():
    input_source_code = row["trimmed_concatenated_content"]
    repo_name = row["repo_name"]
    total_token_count = row["total_token_count"]

    payload = {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{prompt}{input_source_code}"
            }
          ]
        }
      ],
      "temperature": 0.0,
      "top_p": 0.95 #,
      #"max_tokens": 800
    }


 
  # Send request to Open AI to generate readmes with prompt
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

        json_string = response.text
        json_dict = json.loads(json_string)
        output = json_dict["choices"][0]["message"]["content"]
        results.append((repo_name,prompt,total_token_count,output))
        print(output)

    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    results_df = spark.createDataFrame(results,["repo_name","prompt","repo_token_count","virtual_readme"])
    results_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{output_table_path}")


    

# COMMAND ----------

Call_OpenAI()
