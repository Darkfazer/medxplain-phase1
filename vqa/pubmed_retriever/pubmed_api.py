import os
import requests
import xml.etree.ElementTree as ET
import json
import time

CACHE_FILE = "pubmed_cache.json"

def search_pubmed(query, max_results=5):
    """Searches PubMed using NCBI E-utilities, parses XML, and caches results."""
    # Check cache first
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                pass
                
    current_time = time.time()
    cache_key = f"{query}_{max_results}"
    
    # Cache > 24 hours = 86400 seconds
    if cache_key in cache:
        cached_entry = cache[cache_key]
        if current_time - cached_entry['timestamp'] < 86400:
            print(f"Using cached PubMed results for query: {query}")
            return cached_entry['results']
            
    try:
        print(f"Querying NCBI PubMed for: {query}")
        # 1. Search for PMIDs
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
        search_resp = requests.get(search_url, timeout=10)
        search_data = search_resp.json()
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return []
            
        # 2. Fetch Abstracts
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
        fetch_resp = requests.get(fetch_url, timeout=10)
        
        root = ET.fromstring(fetch_resp.content)
        results = []
        
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle')
            title_text = title.text if title is not None else "No Title"
            
            abstract_texts = article.findall('.//AbstractText')
            abstract_text = " ".join([t.text for t in abstract_texts if t.text is not None])
            
            author_list = article.find('.//AuthorList')
            author_name = "Unknown"
            if author_list is not None and len(author_list) > 0:
                last_name = author_list[0].find('LastName')
                if last_name is not None:
                    author_name = last_name.text
                    
            year = "Unknown"
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                yr = pub_date.find('Year')
                if yr is not None:
                    year = yr.text
                    
            if abstract_text:
                results.append({
                    'pmid': pmid,
                    'title': title_text,
                    'abstract': abstract_text,
                    'author': author_name,
                    'year': year
                })
                
        # Update Cache
        cache[cache_key] = {
            'timestamp': current_time,
            'results': results
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
            
        return results
        
    except Exception as e:
        print(f"PubMed API Error: {e}")
        return []
