# -*- coding: utf-8 -*-

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

class WikiEval:
    def __init__(self, client, pre_load=False):
        self.client = client
        self.model_embeddings = SentenceTransformer("sergeyzh/BERTA")
        self.root_dir = r"Articles\Sources"
        self.bm_dir = r'Generation\Utils\bm25_index'
        self.dataset_dir = r'Generation\Utils\text_corpus\snippets.pkl'
        self.embeddings_dir = 'Generation/Utils/embeddings/'
        ru_stopwords = stopwords.words("russian")
        en_stopwords = stopwords.words("english")
        self.combined_stopwords = set(ru_stopwords + en_stopwords)
        self.russian_stemmer = Stemmer.Stemmer("russian")
        self.english_stemmer = Stemmer.Stemmer("english")
        self.pre_load = pre_load
        if pre_load:
            self.retriever = bm25s.BM25.load(self.bm_dir, load_corpus=False)
            self.snippets = self.get_texts_from_disk()
        else:
            self.snippets = self.load_corpus()

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
    
    def ultra_stemmer(self, words_list):
        return [
            self.russian_stemmer.stemWord(word) if self.is_cyrillic(word) else self.english_stemmer.stemWord(word)
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

    def tokenize_query(self, query: str):
        return bm25s.tokenize(query, stopwords=self.combined_stopwords, stemmer=self.ultra_stemmer)
    
    def dcg(self, relevances, k):
        return sum(rel / math.log2(i+2) for i, rel in enumerate(relevances[:k]))
    
    def ndcg(self, relevances, k=None):
        if k is None:
            k = len(relevances)
        dcg_score = self.dcg(relevances, k)
        sorted_relevances = sorted(relevances, reverse=True)
        idcg = self.dcg(sorted_relevances, k)
        return (dcg_score / idcg) if idcg > 0 else 0.0
    
    def r_precision(self, relevances):
        r = sum(relevances)
        return sum(relevances[:int(r)])/r if r else 0.0
    
    async def rank_one_query(self, query: str, top_k: int, true_name: str):
        '''
        Оценивает способность LLM ранжировать выдачу под конкрентную статью
        '''
        tokenized_query = self.tokenize_query(query)
        results, scores = self.retriever.retrieve(tokenized_query, k=top_k)
        source_names = []
        snippets_list = list(self.snippets.items())
        chosen_snippets = [results[0, i] for i in range(results.shape[1])]
        source_texts = [snippets_list[idx][1] for idx in chosen_snippets]
        source_names = [snippets_list[idx][0][0].split("\\")[-2] for idx in chosen_snippets] # другой вид сохранения сниппетов!
        probs = await self.client.filter_sources(true_name, source_texts)
        ranking = [(prob[1], name) for name, prob in zip(source_names, probs)]
        ranking = sorted(ranking, reverse=True)
        relevances = [1 if name == true_name else 0 for _, name in ranking]
        ndcg_score = self.ndcg(relevances, top_k)
        r_pr_score = self.r_precision(relevances)
        return ndcg_score, r_pr_score

    def append_neighbors(self, chosen_snippet, neighbor_count=2):
        # doc_id = Articles\Sources\Article_name\source_№.txt
        # snippet = (name, doc_num, snippet_num) : text
        snippet_num = chosen_snippet[2]
        same_snippets = [(sn[0][2], sn[1]) for sn in self.snippets.items() if sn[0][0] == chosen_snippet[0] and sn[0][1] == chosen_snippet[1]]
        start_idx = snippet_num - neighbor_count
        end_idx = snippet_num + neighbor_count
        text = []
        for s in same_snippets:
            if s[0] >= start_idx and s[0] <= end_idx:
                text.append(s[1])
        combined_snippet = " ".join(txt for txt in text)
        return combined_snippet

    async def rank_outline(self, name, true_heads, neighbor_count=0):
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        snippets_filtered = [sn[1] for sn in self.snippets.items() if sn[0][0] == name]
        snippets_id = [sn[0] for sn in self.snippets.items() if sn[0][0] == name]
        if self.pre_load:
            path = self.embeddings_dir + name + '.pkl'
            with open(path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            os.makedirs(self.embeddings_dir, exist_ok=True)
            embeddings = self.model_embeddings.encode(snippets_filtered)
            file_path = os.path.join(save_dir, f"{name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(embeddings, f)
        silhouette_avg = []
        for num_clusters in list(range(2,20)):
            kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init = 10)
            kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            silhouette_avg.append(score)
        kluster_num = np.argmax(silhouette_avg)+2
        kmeans = KMeans(n_clusters=kluster_num)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        clusters = {}
        for label, snippet_id in zip(labels, snippets_id):
            clusters.setdefault(label, []).append(snippet_id)
        collected_snippets = {}
        for label, snippet_id in clusters.items():
            chosen_snippets = [random.sample(snippet_id, min(3, len(snippet_id))) for _ in range(3)]
            sample_snippets = ["\n".join([self.append_neighbors(chosen_snippet_id, neighbor_count) for chosen_snippet_id in chosen_snippet]) for chosen_snippet in chosen_snippets]
            collected_snippets.setdefault(label, sample_snippets)
        outline = await self.client.generate_outline(collected_snippets, name)
        print(outline)
        parsed_headings = []
        for line in outline.split('\n'):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                title = line[level:].strip()
                if title:
                    parsed_headings.append((level, title))
        heads = [head[1] for head in parsed_headings]
        pred_emb = model.encode(heads, normalize_embeddings=True)
        ref_emb =  model.encode(true_heads, normalize_embeddings=True)
        sims = pred_emb @ ref_emb.T
        precision_scores = sims.max(axis=1).mean()
        recall_scores = sims.max(axis=0).mean()
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision_scores, recall_scores, f1
