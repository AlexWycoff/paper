import requests
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
import os

def semantic_scholar_query(query, limit):
    ''' Takes an integer limit and a string query where space is replaced with +, and = &, or = |. 
    Returns a Pandas DataFrame with paper titles, ids, abstracts, tldrs, and fields. '''
    paper_df = pd.DataFrame()
    
    r = {'message': ''}
    while list(r.keys())[0] == 'message':
        query = 'wind+speed&prediction|predict'
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

def core_query(query, limit):
    ''' Takes an integer limit and a string query where space is replaced with +, and = AND, or = OR. 
    Returns a Pandas DataFrame with paper titles, abstracts, links, and text. '''
    paper_df = pd.DataFrame()
    
    core_api_key = '0IyesnpxHglJ8brOMXzUQfNi91S7PR3Z'
    headers={"Authorization" : "Bearer " + core_api_key}
    url = 'https://api.core.ac.uk/v3/search/works/'
    r = requests.get(url + '?q=' + query + f'&limit={limit}', headers=headers).json()['results']
    
    paper_df['title'] = [r[i]['title'] for i in range(len(r))]
    paper_df['abstract'] = [r[i]['abstract'] for i in range(len(r))]
    paper_df['link'] = [r[i]['downloadUrl'] for i in range(len(r))]
    
    text_list = []
    for url in paper_df['link']:
        if 'abs' in url:
            url_left = url[:url.find('abs')] 
            url_right = url[url.find('abs') + 3:]
            url = url_left + 'pdf' + url_right
        
        text = ""
        try:
            r = requests.get(url)
            filename = Path('temp.pdf')
            filename.write_bytes(r.content)
            reader = PdfReader("temp.pdf")
            for i in range(len(reader.pages)):
                page = reader.pages[i]
                text += page.extract_text()
        except:
            print(f'Bad link: {url}')
        
        # Remove links and newlines from the text (to reduce noise)
        text = text.replace("\n", "")
        new_text = []
        for word in text.split(" "):
            word = 'http' if word.startswith('http') else word
            new_text.append(word)
        text = " ".join(new_text)
        text_list.append(text)
        
    paper_df['text'] = text_list
    os.remove('temp.pdf')
    return paper_df