from vqa.pubmed_retriever.pubmed_api import search_pubmed
from vqa.pubmed_retriever.embedding_cache import PubMedEmbeddingCache

# Initialize embedder globally so model is cached
embeder = None

def get_embedder():
    global embeder
    if embeder is None:
        try:
            embeder = PubMedEmbeddingCache()
        except ImportError:
            pass
    return embeder

def build_context(question, pathologies):
    """
    Take question + image findings
    Extract keywords
    Build search query
    Format retrieved papers as: "According to [Author] (Year) [PMID]: [finding]"
    """
    pathology_str = " ".join([p for p in pathologies if p.lower() != "none" and float(pathologies[p]) > 0.5])
    if not pathology_str:
        pathology_str = "normal"
        
    stop_words = ["what", "is", "the", "in", "this", "image", "are", "there", "any", "shown", "a", "an", "?"]
    clean_q = " ".join([w for w in question.lower().split() if w not in stop_words])
    
    search_term = f"{pathology_str} {clean_q}".strip().replace(" ", "+")
    
    # Try not to overwhelm API if too empty
    if not search_term:
        search_term = "radiography"
        
    print(f"Building PubMed Context with query: {search_term}")
    raw_results = search_pubmed(search_term, max_results=10)
    
    query_context = f"{question} {pathology_str}"
    
    emb = get_embedder()
    if emb is not None:
        best_results = emb.search_similar(query_context, raw_results, top_k=3)
    else:
        best_results = raw_results[:3]
    
    formatted_citations = []
    for res in best_results:
        abstract = res['abstract']
        # Naive first sentence extraction
        snippet = abstract.split('.')[0] + "." 
        
        citation = f"According to {res['author']} et al. ({res['year']}) [PMID: {res['pmid']}]: {snippet}"
        formatted_citations.append(citation)
        
    context_str = "\n".join(formatted_citations)
    if not context_str:
        context_str = "No highly relevant literature found."
        
    return context_str
