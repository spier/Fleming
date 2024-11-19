# Create and Register the Model

# Documentation

After the corpus.txt file has been created it is now possible to load the corpus to an open source semantic search model and register the model with databricks. Project Fleming embraces open-source and can be used with any open-source model on Hugging Face.

Please find an example below.

For more information about options within the Class please follow the documentation under the [code-reference](../code-reference/ModelTrainRegister.md) section.

# Example

```python  

from fleming.discovery.model_train_register import ModelTrainRegister, SemanticSearchModel    
from pyspark.sql import SparkSession

# Not required if using Databricks
spark = SparkSession.builder.appName("model_serving").getOrCreate()

model_directory = "/tmp/BERT_Semantic_Search_model"
corpus_file = "/tmp/search_corpus.txt"
corpus_embedding_file = '/tmp/corpus_embedding.pt'

model_developer = ModelTrainRegister(spark, model_directory, corpus_file, corpus_embedding_file)

# Register the model
semantic_search_model = "multi-qa-mpnet-base-dot-v1"
model_developer.register_model(semantic_search_model)

# Embed the corpus
model_developer.embed_corpus()

# Define parameters and artifacts
parameters = {"top_k": 50, "relevancy_score": 0.45}
input_example = ["Innersource best practices"]
test_output = ["match 1", "match 2"]
signature = infer_signature(input_example, test_output, params=parameters)
artifacts = {
    "model_path": model_directory,
    "corpus_file": corpus_file,
    "corpus_embedding_file": corpus_embedding_file
}
unique_model_name = "semantic_search_model"

# Create and serve the model
experiment_location = "/path/to/experiment"
model_developer.create_registered_model(unique_model_name, input_example, signature, artifacts, experiment_location)
```
