# -*- coding: utf-8 -*-

import requests
import re
import json
from newspaper import Article
from goose3 import Goose
from wiki_parse import WikiParser
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# min text length
MIN_THRESHOLD = 1500

class Extractor(WikiParser):
    def __init__(
        self, 
        article_name: str = '', 
        is_links_downloaded: bool = False, 
        verbose: bool = True, 
        is_downloaded: bool = False, 
        needs_saving: bool = True
    ) -> None:
        super().__init__(
            article_name=article_name, 
            verbose=verbose, 
            is_downloaded=is_downloaded, 
            needs_saving=needs_saving
        )

        self.verbose = verbose
        self.needs_saving = needs_saving
        self.save_dir = self.main_dir / 'Downloaded_Sources_List'
        self.source_texts_path = self.main_dir / 'Sources'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_links_dir = self.save_dir / f'{self.cleared_name.replace(' ', '_')}.json'
        self.downloaded_links = None
        
        if is_links_downloaded:
            with self.downloaded_links_dir.open('r', encoding='utf-8') as f:
                self.downloaded_links = json.load(f)
            
        self.references_positions = None
        self.source_html_texts = None
        self.ref_texts = None
        self.links_num = None
        self.filtered_outline = None
        self.filtered_text = None

    def fetch_article_text(self, entries: list[tuple[str, str]]) -> str:
        """
        Downloads an article using newspaper3k and goose
        """
        for url, label in entries:
            if not url.startswith('http'):
                continue

            text = None
            try:
                # trying to download ru article
                article = Article(url, language='ru')
                article.download()
                article.parse()
                text = article.text
                
                if not text:
                    # trying to download en article
                    article = Article(url, language='en')
                    article.download()
                    article.parse()
                    text = article.text
                    
                if not text:
                    # trying to download with goose
                    g = Goose({'target_language':'ru'})
                    article = g.extract(url=url)
                    text = article.cleaned_text
                    
                if text:
                    return text
                    
            except:
                continue
        return None
    
    def save_text_to_file(self, text: str, filename: str, article_name: str) -> None:
        safe_name = re.sub(r'[<>:"/\\|?*]', '', article_name)
        text_directory = self.source_texts_path / safe_name
        text_directory.mkdir(parents=True, exist_ok=True)
        filepath = text_directory / filename
        with filepath.open('w', encoding='utf-8') as f:
            f.write(text)
    
    
    def fast_extract(self) -> None:
        '''
        Fast download with ThreadPoolExecutor
        '''
        if self.links is None:
            raise ValueError('Error: self.links is None. Call WikiParser.get_references() first.')
        
        max_workers = min(20, len(self.links.keys()))
        successful_urls = []
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.fetch_article_text, entries): ref_id 
                for ref_id, entries in self.links.items()
            }

            idx = 0
            for future in tqdm(
                as_completed(future_to_url), 
                total=len(self.links), 
                desc="Retrieving sources", 
                disable=not self.verbose
            ):
                ref_id = future_to_url[future]
                try:
                    text = future.result()
                    if text and len(text) > MIN_THRESHOLD:
                        idx += 1
                        filename = f'source_{idx}.txt'
                        self.save_text_to_file(text, filename, self.name)
                        successful_urls.append(ref_id)
                except:
                    continue
                    
        if self.needs_saving:
            with self.downloaded_links_dir.open('w', encoding='utf-8') as f:
                json.dump(successful_urls, f, ensure_ascii=False, indent=2)

        self.downloaded_links = successful_urls

    def get_reference_positions(self):
        '''Getting positions which are referenced in text'''
        if self.outline is None:
            raise ValueError('Error: self.outline is None. Call WikiParse.get_outline() first.')
        
        if self.links is None:
            raise ValueError('Error: self.links is None. Call WikiParser.get_references() first.')
            
        link_num = {}
        texts = (self.parser).find_all("div", attrs={"class":"mw-parser-output"})
        for ref_id in tqdm(
            self.links.keys(), 
            desc="Getting link numbers", 
            disable=not self.verbose
        ):
            for ref in texts:
                all_link =  ref.find_all("sup")
                for link in all_link:
                    links_sup = link.find("a")
                    if not links_sup:
                        continue
                        
                    href = links_sup.get('href')
                    if not href or href[1:] != ref_id:
                        continue
                    
                    number_match = re.search(r'\[(\d+)\]', str(links_sup))
                    if not number_match:
                        continue
                        
                    number_text = number_match.group(0)
                    number_inner = re.search(r'\d+', number_text)
                    if not number_inner:
                        continue
                    number_inner_text = number_inner.group(0)
                    link_num[number_inner_text] = ref_id
                  
        references_positions = {}
        for header, section_text in tqdm(
            (self.outline).items(), 
            desc="Calculating reference positions", 
            disable=not self.verbose
        ):
            paragraphs = section_text.split("\n\n")
            for idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                matches = re.findall(r'\[(\d+)\]', paragraph)
                for citation_num in matches:
                    reference_id = link_num.get(citation_num)
                    if not reference_id:
                        continue

                    position = (header, idx + 1)
                    references_positions.setdefault(reference_id, []).append(position)

        self.links_num = link_num
        self.references_positions = references_positions
        return references_positions

    def invert_dict(self, d):
        inverted = {}
        for ref_id, positions in d.items():
            for position in positions:
                inverted.setdefault(position, []).append(ref_id)
        return inverted

    def get_filtered_outline(self):
        '''Clearing text that isn't supported by any source'''
        if self.outline is None:
            raise ValueError("Error: self.outline is None. Call WikiParse.get_outline() first.")

        if self.ref_texts is None and self.downloaded_links is None:
            raise ValueError("Error: sources haven't been downloaded yet!")
        
        ref_pos = self.get_reference_positions()
        inverted_ref = self.invert_dict(ref_pos)
        filtered_outline = {}
        
        for header, section_text in self.outline.items():
            if not section_text:
                filtered_outline[header] = ''
                continue
                
            new_text_parts = []
            paragraphs = section_text.split("\n\n")
            for idx, paragraph in enumerate(paragraphs):
                position = (header, idx + 1)
                if position not in inverted_ref:
                    continue

                keep_paragraph = False
                for source in inverted_ref[position]:
                    if (
                        self.ref_texts 
                        and source in self.ref_texts 
                        and self.ref_texts[source]
                    ) or (
                        self.downloaded_links 
                        and source in self.downloaded_links
                    ):
                        keep_paragraph = True
                        break
                        
                if keep_paragraph:
                    clean_paragraph = re.sub(r'\[(\d+)\]', '', paragraph).strip()
                    if clean_paragraph:
                        new_text_parts.append(clean_paragraph)

            new_text = '\n\n'.join(new_text_parts)
            if new_text or header[0] == 'h2': # always save h2 sections
                filtered_outline[header] = new_text

        self.filtered_outline = filtered_outline

    def get_filtered_text(self):
        '''Returns article text cleared from information that was not supported by any source'''
        if self.filtered_outline is None:
            raise ValueError("Error: self.filtered_outline is None. Call get_filtered_outline() first.")

        article_filtered_text_parts = []
        for (level, title), uncleared_text in self.filtered_outline.items():
            text = re.sub(r'\s+\[(\d+)\]', '', value).strip()
            text = re.sub(r'\[(\d+)\]', '', text).strip()
            article_filtered_text_parts.append(title)
            article_filtered_text_parts.append(text)
            article_filtered_text_parts.append('')

        article_filtered_text = '\n'.join(article_filtered_text_parts)
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
                
        self.source_html_texts = links_by_sources
        