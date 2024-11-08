# paper: AI for exploratory research

The goal of this project is to use retrieval-augmented generation to help students understand the limitations of current research in fields they are interested in and provide them with open-access research from those fields.

## Requirements

- Pandas
- Requests
- Azure AI libraries and API key

## Notebooks:

- Exploratory Research: This notebook demonstrates an example of using retrieval augmented generation to find papers and generate limitations.
- Core & Semantic Scholar: develops functions for retrieving, cleaning, and sorting data from the CORE and Semantic Scholar APIs.
- Researching: Compares different methods of hosting the Phi-3.5 model and includes functions used to query the model
- Querying: Develops the method for creating boolean search queries with Phi-3.5 and using them to search Semantic Scholar and CORE.
- functions: python file with all of the functions developed throughout the above notebooks.

## Data:

We use the Semantic Scholar API to retrieve paper abstracts and metadata from their database and the CORE API to retrieve full text of open access papers and paper metadata.
