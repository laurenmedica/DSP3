## Data Source
The dataset was obtained from Hugging Face and is not originally hosted within this repository.

## Storage Note
Due to the large size of the dataset, it is not stored directly in this GitHub repository. GitHub has strict file size limits, and uploading the full dataset would exceed those constraints.

## Accessing the Data
Instead of storing the files here, the dataset should be downloaded programmatically or accessed using the provided Hugging Face dataset pipeline.

Example (Python):

```python
from datasets import load_dataset

ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
