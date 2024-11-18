
# Query the Endpoint

# Documentation

Once you have your model serving on a databricks endpoint it is then possible to query this data.

Please find an example below.

For more information about options within the Class please follow the documentation under the code-reference section.

# Example

```python

url = "https://example.com/model_endpoint"
token = "your_auth_token"
    
# Create an instance of ModelQuery
model_query = ModelQuery(url, token)
    
# Example dataset
dataset = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    
try:
    # Score the model using the dataset
    response = model_query.score_model(dataset)
    print(response)
except requests.exceptions.HTTPError as e:
    print(f"Error: {str(e)}")

```


[def]: /Users/amber.rigg/Projects/Fleming/docs/code-reference/ModelQuery.md