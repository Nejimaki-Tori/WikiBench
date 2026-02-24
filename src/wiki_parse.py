# -*- coding: utf-8 -*-

from pathlib import Path
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import time
from typing import Dict, List, Tuple
from bs4 import Tag
import base64
import urllib.parse


try:
    repo_root = Path(__file__).resolve().parents[1]
except NameError:
    repo_root = Path.cwd().resolve()
    
class WikiParser:
    """
    Class for parsing HTML pages from RuWiki
    """
    def __init__(
        self, 
        article_name: str,
        verbose: bool = True, 
        is_downloaded: bool = False, 
        needs_saving: bool = True,
        main_dir: str = 'Articles'
    ) -> None:
        self.name = article_name
        self.cleared_name = re.sub(r'[<>:"/\\|?*]', '', article_name)
        self.link = ('https://ru.ruwiki.ru/wiki/' + self.name).replace(' ', '_')
        
        self.main_dir = Path(repo_root) / main_dir
        self.main_dir.mkdir(exist_ok=True)
        self.html_path = self.main_dir / 'Html'
        self.html_path.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        self.html_text = ''
        
        if is_downloaded:
            path_for_html_texts = self.html_path / f'{self.cleared_name}.html'
            with path_for_html_texts.open('r', encoding='utf-8') as f:
                self.html_text = f.read()
        else:
            self.html_text = self.download_article(needs_saving=needs_saving)
            
        self.parser = BeautifulSoup(self.html_text, 'html.parser')
        self.outline = None
        self.text = ''
        self.links = None

    def download_article(self, needs_saving: bool = True) -> str:
        '''Function for downloading HTML from RuWiki'''
        max_attempts = 5
        headers = {
            "User-Agent": ( 
                'Mozilla/5.0 (X11; CrOS x86_64 12871.102.0)' 
                'AppleWebKit/537.36 (KHTML, like Gecko)' 
                'Chrome/81.0.4044.141 Safari/537.36'
            )
        }
        for attempt in range(1, max_attempts + 1):
            try:
                self.response = requests.get(
                    self.link,
                    headers=headers,
                    stream=True,
                    allow_redirects=False,
                    timeout=(3, 7)
                )
                (self.response).raise_for_status()
                html_text = self.response.text
                
                if needs_saving:
                    path_for_article_html_text = self.html_path / f'{self.cleared_name}.html'
                    with path_for_article_html_text.open('w', encoding='utf-8') as f:
                        f.write(html_text)
                        
                return html_text
                
            except requests.RequestException as e:
                if attempt == max_attempts:
                    raise e 
                else:
                    if self.verbose:
                        print('Trying again!')
                    time.sleep(1)     
    
    def decode_external_gateway_href(self, href: str):
        if not href.startswith("#"):
            return None
    
        token = href[1:]
    
        for offset in range(0, 8):
            encoded = token[offset:]
            if not encoded:
                continue

            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                decoded_bytes = base64.b64decode(padded)
                decoded = decoded_bytes.decode("utf-8", errors="ignore")
            except Exception:
                continue

            pos = decoded.find("http")
            if pos == -1:
                continue
    
            candidate = decoded[pos:]
            url = urllib.parse.unquote(candidate)
            return url
    
        return None
    
    def get_references(self) -> dict[str, list[tuple[str, str]]]:
        """Downloading article source-links."""
        references = self.parser.find_all("div", attrs={"class": "mw-references-wrap"})
        if not references:
            references = self.parser.find_all("div", attrs={"class": "mw-page-container-inner-box"})
    
        links_by_citations = {}
    
        for reference_section in references:
            all_ref = reference_section.find_all("li")
    
            for r in all_ref:
                ref_content = []
                ref_id = r.get("id")
                if not ref_id:
                    continue
    
                reference_text = r.find("span", class_="reference-text")
                if reference_text:
                    for a in reference_text.find_all("a", href=True):
                        href: str = a["href"]
                        label: str = a.get_text(strip=True)
                        classes = a.get("class", [])

                        if "external-gateway" in classes:
                            decoded = self.decode_external_gateway_href(href)
                            if decoded:
                                ref_content.append((decoded, label))
                                continue

                        if href.startswith("//"):
                            href = "https:" + href
                        if href.startswith("http"):
                            ref_content.append((href, label))

                if not ref_content:
                    reference_text = r.find("span", class_="reference-text")
                    if reference_text:
                        internal_link = reference_text.find("a", href=True)
                        if not internal_link:
                            text = reference_text.get_text(strip=True)
                            if text:
                                ref_content.append((text,))
                        else:
                            target_id = internal_link["href"].lstrip("#")
                            literature_span = self.parser.find(
                                "span",
                                {"id": target_id}
                            )
                            if literature_span:
                                lit_links = literature_span.find_all("a", href=True)
                                if lit_links:
                                    for lnk in lit_links:
                                        lhref: str = lnk["href"]
                                        llabel: str = lnk.get_text(strip=True)
                                        if lhref.startswith("//"):
                                            lhref = "https:" + lhref
                                        if lhref.startswith("http"):
                                            ref_content.append((lhref, llabel))
                                else:
                                    text = literature_span.get_text(strip=True)
                                    if text:
                                        ref_content.append((text,))
    
                if ref_content:
                    links_by_citations[ref_id] = ref_content
    
        self.links = links_by_citations
        return links_by_citations

    
    def get_outline(self):
        '''Returns an oultline for an article'''
        outline = {}
        current_level = "h1"
        current_title = self.name
        current_text_list = []
        content_div = (self.parser).find('div', {'class': 'mw-parser-output'})
        if content_div:
            elements = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not elements:
                content_div = (self.parser).find('div', {'class': 'mw-page-container-inner-box'})
                elements = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if not elements:
                    print("Error! Wrong page format!")
                    return
            for el in tqdm(elements, desc="Creating outline", disable=not self.verbose):
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
        
        
    def get_text(self) -> str:
        '''Returns full article text'''
        article_text = ""
        for key, value in (self.outline).items():
            text = re.sub(r'\s+\[(\d+)\]', '', value).strip()
            text = re.sub(r'\[(\d+)\]', '', text).strip()
            article_text += key[1] + '\n' + text + '\n'
        self.text = article_text
        return article_text