import os
from nltk.corpus import stopwords
import bm25s
import Stemmer
import re
import pickle
from pymorphy3 import MorphAnalyzer
from pathlib import Path
from wiki_extract import Extractor
from dataclasses import dataclass


CYRILLIC_RE = re.compile(r'[А-Яа-яЁё]')

@dataclass(frozen=True)
class SnippetKey:
    '''Key for getting snippet from the corpus'''
    article_name: str      # article name (folder)
    source_id: int    # source number (in article)
    snippet_id: int   # snippet number (in source document)

# --- MAIN UTILITY CLASS ---

class WikiUtils:
    '''
    Class for working with file structures and collecting snippets, bm25 and embeddings info.
    Also has functions that work with RuWiki articles.
    '''
    def __init__(
        self, 
        device=None, 
        encoder=None,
        root_dir: str = 'Articles', # main folder for articles
        util_dir: str = 'Utils' # main folder for utility inforamtion (snippets, embeddings, etc.)
    ):
        self.device = device
        self.encoder = encoder
        
        self.root_dir = Path(root_dir)
        self.util_dir = Path(util_dir)
        
        self.bm_dir = self.util_dir / 'bm25_index'
        self.bm_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_folder = self.util_dir / 'text_corpus'
        self.dataset_folder.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.dataset_folder / 'snippets.pkl'

        self.embeddings_dir = self.util_dir / 'embeddings'
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir = self.util_dir / 'annotations'
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        self.combined_stopwords = None
        self.english_stemmer = None
        self.morph = None
        self.bm25 = None
        self.snippets = None

    # --- Corpus managment block ---
    
    def create_article_corpus_from_scratch(self, article_bare_names: list[str] = None):
        '''Downloads all articles and their sources'''
        for article_name in article_bare_names: # article names are not cleared from special symbols
            self.get_article(article_name, retrieve_sources=True, verbose=True)

    def create_corpus_from_scratch(self, article_names: list[str] = None, window_size: int = 600, overlap: int = 0):
        '''Creates corpus of snippets, bm25 and precomputed embeddings'''
        ru_stopwords = stopwords.words('russian')
        en_stopwords = stopwords.words('english')
        self.combined_stopwords = set(ru_stopwords + en_stopwords)

        self.english_stemmer = Stemmer.Stemmer('english')
        self.morph = MorphAnalyzer(lang='ru')

        self.build_snippets_from_disk(window_size=window_size, overlap=overlap, needs_saving=True)
        self.build_bm25_index(needs_saving=True)

        for article_name in article_names:
            article_cleared_name = re.sub(r'[<>:"/\\|?*]', '', article_name)
            self.get_embeddings(article_cleared_name=article_cleared_name, force_recompute=True)        

    def load_created_corpus(self):
        '''Loads corpus if it is already prepared'''
        ru_stopwords = stopwords.words('russian')
        en_stopwords = stopwords.words('english')
        self.combined_stopwords = set(ru_stopwords + en_stopwords)

        self.english_stemmer = Stemmer.Stemmer('english')
        self.morph = MorphAnalyzer(lang='ru')

        self.load_snippets_from_disk()
        self.load_bm25_from_disk()

    # --- Texts and snippets block ---
            
    def load_texts_from_directory(self, pattern='*.txt'):
        """Recursive collection of all .txt files"""
        texts = []
        for path in self.root_dir.rglob(pattern):
            if path.is_file():
                texts.append((path, path.read_text(encoding='utf-8')))
        return texts
        
    def chunk_text(self, text: str, window_size: int = 300, overlap: int = 0):
        '''Splitting text into snippets'''
        if window_size <= 0:
            raise ValueError('Window size must be > 0')
            
        if overlap < 0 or overlap >= window_size:
            raise ValueError('Overlap must be in [0, window_size)')
            
        words = text.split()
        snippets = []
        i = 0
        while i < len(words):
            snippet = " ".join(words[i : i + window_size])
            snippets.append(snippet)
            if i + window_size >= len(words):
                break
            i += window_size - overlap
        return snippets

    def is_cyrillic(self, word: str) -> bool: # word is considered russian if it has one russian letter in it
        return bool(CYRILLIC_RE.search(word))

    def word_lemmatize(self, word):
        if self.morph is None:
            raise ValueError('Morph has not been created!')
            
        return self.morph.parse(word)[0].normal_form
        
    def ultra_stemmer(self, words):
        if self.english_stemmer is None:
            raise ValueError('Stemmer for english language has not been created!')
            
        result = []
        for word in words:
            if self.is_cyrillic(word):
                result.append(self.word_lemmatize(word))
            else:
                result.append(self.english_stemmer.stemWord(word))
        
        return result

    def get_number_of_snippets(self):
        '''Returns number of snippets per article'''
        counts = {}
        for snippet_key in self.snippets.keys():
            counts[snippet_key.article_name] = counts.get(snippet_key.article_name, 0) + 1
            
        return counts

    def parse_source_id_from_filename(self, path) -> int:
        stem = path.stem  # 'source_1.txt' -> 'source_1'
        match = re.search(r'_(\d+)$', stem)
        if not match:
            raise ValueError(f'Cannot extract source_id from filename: {path.name}')
        return int(match.group(1))
        
    def build_snippets_from_disk(self, window_size: int = 300, overlap: int = 0, needs_saving: bool = False):
        '''Collecting snippets from texts from disk'''
        all_texts = self.load_texts_from_directory(pattern='*.txt')
        snippets = {}
        for file_path, content in all_texts:
            article_name = file_path.parent.name
            source_id = self.parse_source_id_from_filename(file_path)
            file_snippets = self.chunk_text(content, window_size=window_size, overlap=overlap)
            
            for snippet_id, snippet_text in enumerate(file_snippets):
                snippet_key = SnippetKey(article_name=article_name, source_id=source_id, snippet_id=snippet_id)
                snippets[snippet_key] = snippet_text
                
        self.snippets = snippets
        
        if needs_saving:
            self.save_snippets_on_disk()
            
    def save_snippets_on_disk(self, path: str = None) -> None:
        '''Saves all snippets on disk'''
        target_path = self.dataset_path if not path else Path(path)
        with target_path.open('wb') as f:
            pickle.dump(self.snippets, f)
            
    def load_snippets_from_disk(self, path: str = None):
        '''Loads all snippets from disk'''
        target_path = self.dataset_path if not path else Path(path)
        with target_path.open('rb') as f:
            self.snippets = pickle.load(f)

    # --- BM25 block ---

    def build_bm25_index(self, needs_saving: bool = False):
        if self.snippets is None:
            raise ValueError('Snippets have not been created!')

        if self.combined_stopwords is None:
            raise ValueError('Empty stopwords list!')
            
        corpus_texts = list(self.snippets.values())
        
        corpus_tokens = bm25s.tokenize(
            corpus_texts, 
            stopwords=self.combined_stopwords, 
            stemmer=self.ultra_stemmer
        )        

        bm = bm25s.BM25()
        bm.index(corpus_tokens)
        self.bm25 = bm

        if needs_saving:
            self.save_bm25_on_disk(corpus_tokens)

    def save_bm25_on_disk(self, corpus_tokens, path: str = None):
        target_path = self.bm_dir if not path else Path(path)
        self.bm25.save(target_path, corpus=corpus_tokens)
    
    def load_bm25_from_disk(self, path: str = None, load_corpus: bool = False):
        target_path = self.bm_dir if not path else Path(path)
        bm = bm25s.BM25.load(target_path, load_corpus=load_corpus)
        self.bm25 = bm

    def tokenize_query(self, query):
        '''Query tokenesation, same as for the text corpus'''
        if self.bm25 is None:
            raise ValueError('BM25 has not been initialised!')
                    
        return bm25s.tokenize(
            query, 
            show_progress=False, 
            stopwords=self.combined_stopwords, 
            stemmer=self.ultra_stemmer
        )

    # --- Embeddings block ---
    
    def get_embeddings(self, article_cleared_name, force_recompute: bool = False):
        if self.encoder is None:
            raise ValueError('Encoder is not set!')

        embeddings_path = self.embeddings_dir / f'{article_cleared_name}.pkl'
        if not force_recompute:
            return self.load_embeddings_from_disk(embeddings_path=embeddings_path)
                
        selected_snippets = [
            (snippet_key, snippet_text)
            for snippet_key, snippet_text in self.snippets.items() 
            if snippet_key.article_name == article_cleared_name
        ]
        
        ids = [sid for sid, _ in selected_snippets]
        texts = [txt for _, txt in selected_snippets]
    
        embeddings = self.encoder.encode(
            texts, 
            device=self.device
        )
        emb_by_id = {sid: embeddings[i] for i, sid in enumerate(ids)}
        
        self.save_embeddings_on_disk(emb_by_id, embeddings_path=embeddings_path)

        return emb_by_id

    def load_embeddings_from_disk(self, embeddings_path):
        if type(embeddings_path) == str:
            embeddings_path = Path(embeddings_path)
        with embeddings_path.open('rb') as f:
            return pickle.load(f)

    def save_embeddings_on_disk(self, embeddings, embeddings_path):
        if type(embeddings_path) == str:
            embeddings_path = Path(embeddings_path)
        with embeddings_path.open('wb') as f:
            pickle.dump(embeddings, f)

    # --- Article block ---
    
    def get_annotations_from_disk(self):
        texts = []
        for subdir, _, files in os.walk(self.annotation_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        texts.append(text)
        return texts