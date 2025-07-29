# -*- coding: utf-8 -*-

import requests
import re
import json
from newspaper import Article
from goose3 import Goose
from wiki_parse import WikiParser
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# min text length
MIN_THRESHOLD = 1500

class Extracter(WikiParser):
    def __init__(self, article_name: str, downloaded: bool=False, verbose=True, html=False, needs_saving=True):
        super().__init__(article_name, verbose, html, needs_saving)
        self.filtered_outline = None
        self.filtered_text = None
        self.verbose = verbose
        self.needs_saving = needs_saving
        save_dir = os.path.join("Articles", "Downloaded_Sources_List")
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"{self.cleared_name.replace(" ", "_")}.json")
        if downloaded:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.downloaded_links = json.load(f)
        else:
            self.downloaded_links = None
        self.references_positions = None
        self.html_text = None
        self.ref_texts = None
        self.links_num = None

    def fetch_article_text(self, urls: [str]) -> str:
        """
        Загружает и парсит статью по URL с использованием newspaper3k
        """
        for url in urls:
            new_url = url[0]
            text = None
            try:
                # попытка скачать источник на русском языке
                article = Article(new_url, language='ru')
                article.download()
                article.parse()
                text = article.text
                if not text:
                    # попытка скачать источник на английском
                    article = Article(new_url, language='en')
                    article.download()
                    article.parse()
                    text = article.text
                    if not text:
                        # попытка скачать источник другим модулем
                        g = Goose({'target_language':'ru'})
                        article = g.extract(url=url)
                        text = article.cleaned_text
                        if not text:
                            continue
                return text
            except:
                return None
        return None
    
    
    def save_text_to_file(self, text: str, filename: str, article_name: str) -> None:
        article_name = re.sub(r'[<>:"/\\|?*]', '', article_name)
        directory = os.path.join("Articles", "Sources", article_name)
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    
    def fast_extracter(self) -> None:
        '''
        Быстрое скачивание источников с использованием ThreadPoolExecutor
        '''
        max_workers = min(20, len(self.links.keys()))
        successful_urls = []
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_article_text, url): (ref_key, url) for ref_key, url in self.links.items()}

            idx = 0
            for future in tqdm(as_completed(future_to_url), total=len(self.links), desc="Retrieving sources", disable=not self.verbose):
                ref_key, _ = future_to_url[future]
                try:
                    text = future.result()
                    if text and len(text) > MIN_THRESHOLD:
                        idx += 1
                        filename = "source_" + str(idx) + ".txt"
                        self.save_text_to_file(text, filename, self.name)
                        successful_urls.append(ref_key)
                except:
                    pass
        if self.needs_saving:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(successful_urls, f, ensure_ascii=False, indent=2)

        self.downloaded_links = successful_urls

    def get_reference_positions(self):
        """Получение позиций в тексте, на которые ссылаются источники"""
        if self.outline is None:
            raise ValueError("Ошибка: метод get_outline() не был вызван!")
        link_num = {}
        texts = (self.parser).find_all("div", attrs={"class":"mw-parser-output"})
        for item, _ in tqdm((self.links).items(), desc="Getting link numbers", disable=not self.verbose):
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
        for header, section_text in tqdm((self.outline).items(), desc="Calculating reference positions", disable=not self.verbose):
            paragraphs = section_text.split("\n\n")
            for idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                matches = re.findall(r'\[(\d+)\]', paragraph)
                for citation_num in matches:
                    if citation_num in link_num.keys():
                        source = link_num[citation_num]
                        if source not in references_positions:
                            references_positions[source] = [(header, idx + 1)]
                        else:
                            references_positions[source].append((header, idx + 1))
        self.links_num = link_num
        self.references_positions = references_positions
        return references_positions

    def invert_dict(self, d):
        inverted = {}
        for key, value in d.items():
            for val in value:
                if val not in inverted:
                    inverted[val] = [key]
                else:
                    inverted[val].append(key)
        return inverted

    def get_filtered_outline(self):
        '''Удаление текста, не опирающегося на источники'''
        if self.outline is None:
            raise ValueError("Ошибка: метод get_outline() не был вызван!")

        if self.ref_texts is None and self.downloaded_links is None:
            raise ValueError("Ошибка: источники к статье не были скачаны!")
        
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
                        #print(source)
                        if (self.ref_texts and source in self.ref_texts and self.ref_texts[source]) or \
                        (self.downloaded_links and source in self.downloaded_links):
                            #print('Take!: ', source)
                            paragraph = re.sub(r'\[(\d+)\]', '', paragraph).strip()
                            new_text += paragraph + "\n\n"
                            break
            if new_text or header[0] == 'h2':
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

    def html_ref(self):
        '''Сохранение html-кода страниц источников'''
        links_by_sources = {}
        
        for source_key, items in tqdm((self.links).items(), disable=not self.verbose):
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

    # старый способ загрузки источников
    def extract_ref(self):
        '''Сохранение текста источников с достаточным объемом текста'''
        ref_texts = {}

        def fetch_text(url):
            try:
                art = Article(url, language='ru')
                art.download()
                art.parse()
                text = art.text
                if text and len(text) > MIN_THRESHOLD:
                    return text
            except Exception:
                pass
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
        