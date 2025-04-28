import os
import glob
import math
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from datasets import Dataset
import bm25s
import Stemmer
import re
import pickle
from pymorphy3 import MorphAnalyzer

class WikiUtils:
    def __init__(self, pre_load=False):
        self.model_embeddings = SentenceTransformer("sergeyzh/BERTA")
        self.root_dir = r"Articles\Sources"
        self.bm_dir = r'Generation\Utils\bm25_index'
        self.dataset_dir = r'Generation\Utils\text_corpus\snippets.pkl'
        self.embeddings_dir = 'Generation/Utils/embeddings/'
        ru_stopwords = stopwords.words("russian")
        en_stopwords = stopwords.words("english")
        self.combined_stopwords = set(ru_stopwords + en_stopwords)
        #self.russian_stemmer = Stemmer.Stemmer("russian")
        self.english_stemmer = Stemmer.Stemmer("english")
        self.pre_load = pre_load
        self.morph = MorphAnalyzer(lang="ru")
        if pre_load:
            self.retriever = bm25s.BM25.load(self.bm_dir, load_corpus=False)
            self.snippets = self.get_texts_from_disk()
            
    def load_corpus(self, save_bm=None, save_data=None):
        if not save_bm:
            save_bm = self.bm_dir
        if not save_data:
            save_data = self.dataset_dir
        corpus_tokens, snippets = self.tokenize_corpus(save_dataset=save_data)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.retriever.save(save_bm, corpus=corpus_tokens)
        return snippets
        
    def get_texts_from_disk(self, directory=None):
        if not directory:
            directory = self.dataset_dir
        with open(self.dataset_dir, "rb") as f:
            return pickle.load(f)
            
    def load_texts_from_directory(self, file_extension="*.txt"):
        """
        Рекурсивно собираем все файлы с указанным расширением из root_dir.
        Возвращает список (article_path, text_content).
        """
        texts = []
        pattern = os.path.join(self.root_dir, "**", file_extension)
        for filepath in glob.iglob(pattern, recursive=True):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append((filepath, content))
        return texts
        
    def chunk_text(self, text: str, window_size=2000, overlap=512):
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

    def is_cyrillic(self, word):
        return any(c.lower() not in "abcdefghijklmnopqrstuvwxyz" for c in word)

    def word_lemmatize(self, word):
        return self.morph.parse(word)[0].normal_form
        
    def ultra_stemmer(self, words_list):
        return [
            self.word_lemmatize(word) if self.is_cyrillic(word) else self.english_stemmer.stemWord(word)
            for word in words_list
        ]

    def tokenize_corpus(self, window_size=300, overlap=0, save_dataset=None):
        '''
        Токенезация корпуса текстов из коллекции скачанных источников
        '''
        if not save_dataset:
            save_dataset = self.dataset_dir
        all_texts = self.load_texts_from_directory(file_extension="*.txt")
        snippets = {}
        for file_path, content in all_texts:
            file_snippets = self.chunk_text(content, window_size=window_size, overlap=overlap)
            for idx, snippet in enumerate(file_snippets):
                article_name = file_path.split("\\")[-2]
                text_num = int(file_path.split("\\")[-1].split('.')[0].split('_')[1])
                snippets[(article_name, text_num, idx)] = snippet
        
        corpus_texts = [text for text in snippets.values()]
        corpus_tokens = bm25s.tokenize(
            corpus_texts, 
            stopwords=self.combined_stopwords, 
            stemmer=self.ultra_stemmer
        )

        with open(self.dataset_dir, "wb") as f:
            pickle.dump(snippets, f)
        
        return corpus_tokens, snippets

    def tokenize_query(self, query):
        return bm25s.tokenize(query, stopwords=self.combined_stopwords, stemmer=self.ultra_stemmer)

    def get_embeddings(self, name, force_download=False):
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        snippets_filtered = [sn[1] for sn in self.snippets.items() if sn[0][0] == name]
        if self.pre_load and not force_download:
            path = self.embeddings_dir + name + '.pkl'
            with open(path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            embeddings = self.model_embeddings.encode(snippets_filtered)
            file_path = os.path.join(self.embeddings_dir, f"{name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(embeddings, f)
        return embeddings

    def section_to_references(self, page, doc_num):
        downloaded_links = set(list(doc_num.keys()))
        ref_to_sections = {}
        for ref, pos in page.references_positions.items():
        #print(pos[0])
            for p in pos:
                if ref not in ref_to_sections:
                    ref_to_sections[ref] = [p[0]]
                else:
                    ref_to_sections[ref].append(p[0])
        section_to_refs = page.invert_dict(ref_to_sections)
        new_section_to_refs = {}
        for section in section_to_refs.keys():
            chosen_refs = list(set(section_to_refs[section]) & set(downloaded_links))
            if chosen_refs:
                new_section_to_refs[section] = chosen_refs
        return new_section_to_refs

    def section_to_snippets(self, page):
        doc_num = {
                ref_link: doc_id + 1 for doc_id, ref_link in enumerate(page.downloaded_links)
            }
        section_to_refs = self.section_to_references(page, doc_num)
        #print()
        #print(section_to_refs)
        section_to_sn = {}
        for section, refs in section_to_refs.items():
            collected_snippets = []
            #print('finding texts for: ', section)
            #print(refs)
            for ref in refs:
                ref_num = doc_num[ref]
                found_snippets = [((sn_key[0], sn_key[1], sn_key[2]), sn_txt) for sn_key, sn_txt in self.snippets.items() if int(sn_key[1]) == ref_num and sn_key[0] == page.cleared_name]
                found_snippets = sorted(found_snippets, key=lambda x: (x[0][1], x[0][2]))
                #texts_only = [sn[2] for sn in found_snippets]
                collected_snippets += found_snippets
            section_to_sn[section] = collected_snippets
        return section_to_sn