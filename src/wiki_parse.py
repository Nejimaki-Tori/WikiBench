import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import time

class WikiParser:
    def __init__(self, article_name: str):
        """Инициализация парсера с загрузкой HTML-кода."""
        self.name = article_name
        self.headers = {"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}
        self.link = ('https://ru.ruwiki.ru/wiki/' + self.name).replace(" ", "_")
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                self.response = requests.get(
                    self.link,
                    headers=self.headers,
                    stream=True,
                    allow_redirects=False,
                    timeout=(3, 7)
                )
                (self.response).raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_attempts:
                    raise e 
                else:
                    print('Trying again!')
                    time.sleep(1)
        self.parser = BeautifulSoup(self.response.text, 'html.parser')
        self.outline = None
        self.text = ""
        self.links = None

        
    def get_references(self):
        """Выкачка ссылок источников"""
        references = (self.parser).find_all("div", attrs={"class":"mw-references-wrap"})
        links_by_citations = {}
        
        for reference_section in references:
          all_ref =  reference_section.find_all("li")
          for r in all_ref:
            ref_content = []
            ref_id = r.get('id')
            external_links = r.find_all("a", attrs={"class": "external text"})
            if external_links:
              ref_content.extend([(link.get('href'), link.get_text(strip=True)) for link in external_links])
            else:
              reference_text = r.find("span", class_="reference-text")
              if reference_text:
                internal_link = reference_text.find("a", href=True)
                if not internal_link:
                  ref_content.append((reference_text.get_text(strip=True)))
                else:
                   for ul in (self.parser).find_all("ul"):
                      for li in ul.find_all("li"):
                        literature = li.find("span", {"class": "citation no-wikidata", "id": internal_link['href'][1:]})
                        if literature:
                          ref_content.append((literature.get_text(strip=True)))
            if ref_content:
              links_by_citations[ref_id] = ref_content
    
        self.links = links_by_citations


    def get_outline(self):
        '''Получение плана статьи'''
        outline = {}
        current_level = "h1"
        current_title = self.name
        current_text_list = []
        content_div = (self.parser).find('div', {'class': 'mw-parser-output'})
        if content_div:
            elements = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for el in tqdm(elements, desc="Creating outline"):
                if el.name == 'p' and 'ruwiki-universal-dropdown__btn-top' in el.get('class', []):
                    continue
                if el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if current_level is not None and current_title is not None:
                        current_title = re.sub(r'\[.*?\]', '', current_title).strip()
                        outline[(current_level, current_title)] =  "\n\n".join(current_text_list).strip()
                    current_level = el.name
                    current_title = el.get_text()
                    current_text_list = [] 
                else:
                    txt = el.get_text(separator=" ", strip=True)
                    text = re.sub(r'(?<=\S)\s+(?=[,.!?;:])', '', txt)
                    if text:
                        current_text_list.append(text)
                        
            if current_level is not None and current_title is not None:
                current_title = re.sub(r'\[.*?\]', '', current_title).strip()
                outline[(current_level, current_title)] =  "\n\n".join(current_text_list).strip()
                
        items = list(outline.items())
        while items and items[-1][1] == '':
            items.pop()
        outline = dict(items)
        self.outline = outline

        
    def get_text(self):
        '''Получение очищенного текста статьи'''
        article_text = ""
        for key, value in (self.outline).items():
            text = re.sub(r'\s+\[(\d+)\]', '', value).strip()
            text = re.sub(r'\[(\d+)\]', '', text).strip()
            article_text += key[1] + '\n' + text + '\n'
        self.text = article_text
        return article_text