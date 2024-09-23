import requests
import pandas as pd
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

def semantic_scholar_query(query ,limit):
    ''' Takes an integer limit and a string query where space is replaced with +, and = &, or = |. 
    Returns a Pandas DataFrame with paper titles, ids, abstracts, tldrs, and fields. '''
    paper_df = pd.DataFrame()
    
    r = {'message': ''}
    while list(r.keys())[0] == 'message':
        base_url = f'https://api.semanticscholar.org/graph/v1/paper/search'
        r = dict(requests.get(base_url, params={'query': query, 'limit': limit}).json())
    titles = [r['data'][i]['title'] for i in range(len(r['data']))]
    ids = [r['data'][i]['paperId'] for i in range(len(r['data']))]
    
    r = requests.post(
    'https://api.semanticscholar.org/graph/v1/paper/batch',
    params={'fields': 'abstract,tldr,url,s2FieldsOfStudy'},
    json={"ids": ids}).json()
    abstracts = [r[i]['abstract'] for i in range(len(r))]

    tldrs = []
    for i in range(len(r)):
        if r[i]['tldr'] != None:
            tldrs.append(r[i]['tldr']['text'])
        else:
            tldrs.append(r[i]['tldr'])

    fields = [''] * len(r)
    for i in range(len(r)):
        paper_fields = r[i]['s2FieldsOfStudy']
        for j in range(len(paper_fields)):
            fields[i] += paper_fields[j]['category']
            if j != len(paper_fields) - 1:
                fields[i] += ', '
    
    paper_df['title'], paper_df['id'], paper_df['abstract'], paper_df['tldr'], paper_df['field'] = titles, ids, abstracts, tldrs, fields
    return paper_df

def cite_core(results):
    ''' Takes results from querying CORE via core_query for a single paper and returns an incomplete citation in MLA format. '''
    citation = ""
    authors = [author['name'] for author in results['authors']]
    if len(authors) >= 3:
        # If there are more than 3 authors, we use the first author and et al.
        citation += authors[0] + ", et al. "
    elif len(authors) == 2:
        # Get name in First Last order
        comma = authors[1].find(",")
        authors[1] = authors[1][comma + 2:] + " " + authors[1][:comma]
        citation += f"{authors[0]}, and {authors[1]}. "
    else: # len(authors) == 1
        citation += f"{authors[0]}. "
    title = results["title"]
    # Add the title
    if "\n" in title:
        left = title[:title.find("\n")]
        right = title[title.find("\n") + 2:]
        title = left + right
    citation += f'"{title}." '
    # Add the year
    citation += f"{results['yearPublished']}, "
    # Add a link (doi if possible)
    link = results['links'][0]['url']
    if results['doi'] != '' and results['doi'] != None:
        citation += f"https://doi.org/{results['doi']}"
    elif link != '' and link != None:
        citation += link
    return citation

def core_query(query, limit=10, offset=0):
    ''' Takes an integer limit and a string query where space is replaced with +, and = AND, or = OR. 
    Returns a Pandas DataFrame with paper titles, abstracts, links, and text. '''
    paper_df = pd.DataFrame()
    
    core_api_key = '0IyesnpxHglJ8brOMXzUQfNi91S7PR3Z'
    headers={"Authorization" : "Bearer " + core_api_key}
    url = 'https://api.core.ac.uk/v3/search/works/'
    r = requests.get(url + '?q=' + query + f'&limit={limit}' + f'&offset={offset}', headers=headers).json()['results']
    
    paper_df['title'] = [r[i]['title'] for i in range(len(r))]
    paper_df['abstract'] = [r[i]['abstract'] for i in range(len(r))]
    paper_df['link'] = [r[i]['downloadUrl'] for i in range(len(r))]
    paper_df['text'] = [r[i]['fullText'] for i in range(len(r))]
    paper_df['citation'] = [cite_core(result) for result in r]
    
    text_list = []
    for i in range(len(paper_df)):
        # Modify arxiv links to point directly to the article
        url = paper_df.loc[i, 'link']
        if 'abs' in url:
            url_left = url[:url.find('abs')] 
            url_right = url[url.find('abs') + 3:]
            url = url_left + 'pdf' + url_right
        paper_df.loc[i, 'link'] = url
        
        # Remove links and newlines from the text (to reduce noise)
        text = paper_df.loc[i, 'text']
        text = text.replace("\n", "")
        new_text = []
        for word in text.split(" "):
            word = 'http' if word.startswith('http') else word
            new_text.append(word)
        text = " ".join(new_text)
        paper_df.loc[i, 'text'] = text
        
    return paper_df

def api_server_query(context, query, url='http://10.0.0.213:8080'):
    ''' Takes any previous context and a text query and returns AI text output and updated context. '''
    prompt = f"{context}\nUser: {query}\nAssistant:"
    # These variables can be fine-tuned to produce different results.
    # See https://docs.anaconda.com/ai-navigator/user-guide/tutorial/
    data = {
        'prompt': prompt,
        'temperature': 0.8,
        'top_k': 35,
        'top_p': 0.95,
        'n_predict': 400,
        'stop': ["</s>", "Assistant:", "User:"]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f'{url}/completion', json=data, headers=headers)
    if response.status_code == 200:
        context = f"{context}\nUser: {query}\nAssistant: {response}"
        return response.json()['content'].strip(), context
    else:
        return "Error processing your request. Please try again.", context
    
def research_query(query, limit=10, offset=0):
    ''' Takes a text prompt and returns phi 3.5 responses and a dictionary of token usage. '''
    load_dotenv()
    api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
    if not api_key:
      raise Exception("A key should be provided to invoke the endpoint")

    client = ChatCompletionsClient(
        endpoint='https://Phi-3-5-mini-instruct-bzsru.eastus2.models.ai.azure.com',
        credential=AzureKeyCredential(api_key)
    )
    
    paper_df = core_query(query, limit=limit, offset=offset)
    prompt = ""
    for text in paper_df['text']:
        prompt += text
        
    abstract_df = semantic_scholar_query(query, limit)
    for abstract in abstract_df['abstract']:
        if abstract != None:
            prompt += abstract
    
    result = client.complete({
        "messages": [
          {
            "role": "user",
            "content": prompt + "List and explain within the list any potential for future research, centered around no more than 5 major, generalized gaps seen throughout the studies."
          }
        ],
        "temperature": 0.8,
        "top_p": 0.95,
    })
    response = str(result.choices[0].message.content)

    # Bold the text between asterisks so it looks nice
    while "**" in response:
        first = response.find("**")
        left = response[:first]
        second = response[response.find("**") + 2:].find("**") + first + 2
        right = response[second + 2:]
        middle = response[first + 2:second]
        middle = '\033[1m' + middle + '\033[0m'
        response = left + middle + right

    response += "\n\n"
    for i in range(len(paper_df)):
        response += f"    {paper_df.loc[i, 'citation']} \n\n"

    tokens_usage = {"prompt": result.usage.prompt_tokens, 
                    "completion": result.usage.completion_tokens, 
                    "total": result.usage.total_tokens}
    return response, tokens_usage