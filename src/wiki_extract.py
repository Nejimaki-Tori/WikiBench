import requests
import re
from newspaper import Article, fulltext
from wiki_parse import WikiParser
from tqdm import tqdm
from goose3 import Goose

# min text length
MIN_THRESHOLD = 1500

class Extracter(WikiParser):
    def __init__(self, article_name: str):
        super().__init__(article_name)
        self.html_text = None
        self.ref_texts = None
        self.filtered_outline = None
        self.filtered_text = None

    def html_ref(self):
        '''Сохранение html-кода страниц источников'''
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
                            if not extracted_links:
                                extracted_links.append(text)
                    except requests.RequestException as e:
                        continue
            if extracted_links:
                links_by_sources[source_key] = extracted_links
        self.html_text = links_by_sources
    
    
    def extract_ref(self):
        '''Сохранение текста источников с достаточным объемом текста'''
        ref_texts = {}
        max_attempts = 3

        def fetch_text(url):
            for attempt in range(1, max_attempts + 1):
                try:
                    art = Article(url, language='ru')
                    art.download()
                    art.parse()
                    text = art.text
                    if text and len(text) > MIN_THRESHOLD:
                        return text
                except Exception:
                    continue
            try:
                g = Goose({'target_language':'ru'})
                article = g.extract(url=url)
                text = article.cleaned_text
                if text and len(text) > MIN_THRESHOLD:
                    return text
            except Exception:
                return None
            return None
            
        for source_key, items in tqdm(self.links.items(), desc="Retrieving sources"):
            extracted_texts = []
            for item1 in items:
                url = item1[0]
                if url.startswith("http"):
                    text = fetch_text(url)
                    if text:
                        extracted_texts.append(text)
            if extracted_texts:
                ref_texts[source_key] = extracted_texts
                    
        self.ref_texts = ref_texts


    def invert_dict(self, d):
        inverted = {}
        for key, value in d.items():
            if value not in inverted:
                inverted[value] = [key]
            else:
                inverted[value].append(key)
        return inverted
    

    def get_reference_positions(self):
        """Получение позиций в тексте, на которые ссылаются источники"""
        if self.outline is None:
            raise ValueError("Ошибка: метод get_outline() не был вызван!")
        link_num = {}
        texts = (self.parser).find_all("div", attrs={"class":"mw-parser-output"})
        for item, _ in tqdm((self.links).items(), desc="Getting link numbers"):
          for ref in texts:
            all_link =  ref.find_all("sup")
            for link in all_link:
              links_sup = link.find("a")
              if links_sup and links_sup['href'][1:] != item:
                continue
              elif links_sup:
                pattern = r'\[(\d+)\]'
                pattern2 = r'\d+'
                
                number = re.search(pattern, str(links_sup))
                if number:
                    number = number.group(0)
                    number2 = re.search(pattern2, str(number))
                    link_num[number2.group(0)] = item
                else:
                    continue
                  
        references_positions = {}
        for header, section_text in tqdm((self.outline).items(), desc="Calculating reference positions"):
            paragraphs = section_text.split("\n\n")
            for idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                matches = re.findall(r'\[(\d+)\]', paragraph)
                for citation_num in matches:
                    if citation_num in link_num.keys():
                        source = link_num[citation_num]
                        references_positions[source] = (header, idx + 1)

        return references_positions

    
    def get_filtered_outline(self):
        '''Удаление текста, не опирающегося на источники'''
        if self.outline is None:
            raise ValueError("Ошибка: метод get_outline() не был вызван!")

        if self.ref_texts is None:
            raise ValueError("Ошибка: метод newspaper_ref() не был вызван!")
        
        ref_pos = self.get_reference_positions()
        inverted_ref = self.invert_dict(ref_pos)
        filtered_outline = {}
        
        for header, section_text in self.outline.items():
            if not section_text:
                filtered_outline[header] = ''
                continue
            new_text = ""
            paragraphs = section_text.split("\n\n")
            for idx, paragraph in enumerate(paragraphs):
                if (header, idx + 1) in inverted_ref.keys():
                    for source in inverted_ref[(header, idx + 1)]:
                        if source in self.ref_texts and self.ref_texts[source]:
                            new_text += paragraph + "\n\n"
                            break
                else:
                     new_text += paragraph + "\n\n"
            if new_text:
                filtered_outline[header] = new_text

        self.filtered_outline = filtered_outline

    def get_filtered_text(self):
        '''Получение отфильтрованного текста статьи'''
        article_filtered_text = ""
        for key, value in (self.filtered_outline).items():
            text = re.sub(r'\s+\[(\d+)\]', '', value).strip()
            text = re.sub(r'\[(\d+)\]', '', text).strip()
            article_filtered_text += key[1] + '\n' + text + '\n'
        self.filtered_text = article_filtered_text
        return article_filtered_text
        