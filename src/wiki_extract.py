import requests
from newspaper import Article, fulltext
from wiki_parse import WikiParser
from tqdm import tqdm

class Extracter(WikiParser):
    def __init__(self, article_name: str):
        super().__init__(article_name)
        self.html_text = None
        self.ref_texts = None

    def html_ref(self):
        links_by_sources = {}
        
        for source_key, items in tqdm((self.links).items()):
            extracted_links = []
            for item1 in items:
              item = item1[0]
              if item.startswith("http"):
                try:
                  response = requests.get(item, headers=self.headers, stream=True, allow_redirects=False, timeout=(3, 7))
                  response.raise_for_status()
                  text = response.text
                  if text:
                    extracted_links.append(text)
                except requests.RequestException as e:
                  continue
            if extracted_links:
              links_by_sources[source_key] = extracted_links
        self.html_text = links_by_sources
    
    
    def newspaper_ref(self):
        ref_texts = {}
        for source_key, items in tqdm((self.links).items()):
            extracted_links = []
            for item1 in items:
              item = item1[0]
              if item.startswith("http"):
                try:
                  art = Article(item, language='ru')
                  art.download()
                  art.parse()
                  text = art.text
                  if text:
                    extracted_links.append(text)
                except:
                  continue
            if extracted_links:
              ref_texts[source_key] = extracted_links
        self.ref_texts = ref_texts