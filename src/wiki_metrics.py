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
from wiki_utils import WikiUtils

class WikiEval(WikiUtils):
    def __init__(self, client, pre_load=False):
        super().__init__(pre_load=pre_load)
        self.client = client
    
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
        #print('found texts: ', len(text))
        combined_snippet = " ".join(txt for txt in text)
        return combined_snippet

    async def k_means_method(self, name, snippets_filtered, snippets_id, embeddings, neighbor_count, forced_cluster_num, clusters_centers, description_mode=0):
        silhouette_avg = []
        snippet_emb = {}
        for i, sn_id in enumerate(snippets_id):
            snippet_emb[sn_id] = embeddings[i]
        if forced_cluster_num:
            kluster_num = forced_cluster_num
        elif not clusters_centers:
            for num_clusters in list(range(2,20)):
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, kmeans.labels_)
                silhouette_avg.append(score)
            kluster_num = np.argmax(silhouette_avg)+2
        if clusters_centers:
            #print('clust with mega hint')
            kmeans = KMeans(n_clusters=len(clusters_centers), init=clusters_centers, random_state=42)
        else:
            #print('no hint for you!')
            kmeans = KMeans(n_clusters=kluster_num, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        clusters = {}
        for label, snippet_id in zip(labels, snippets_id):
            clusters.setdefault(label, []).append((snippet_id, snippet_emb[snippet_id]))
        collected_snippets = {}
        for label, snippet_id_and_emb in clusters.items():
            cluster_emb = np.array([elem[1] for elem in snippet_id_and_emb])
            if len(cluster_emb) <= 5:
                chosen_snippets = [sn_id[0] for sn_id in snippet_id_and_emb]
            else:
                cluster_center = cluster_emb.mean()
                distances = [(np.linalg.norm(emb[1] - cluster_center), emb[0]) for emb in snippet_id_and_emb]
                distances = sorted(distances)[:5]
                chosen_snippets = [sn_id[1] for sn_id in distances]
            #chosen_snippets = [random.sample(snippet_id, min(3, len(snippet_id))) for _ in range(3)]
            #sample_snippets = ["\n".join([self.append_neighbors(chosen_snippet_id, neighbor_count) for chosen_snippet_id in chosen_snippet]) for chosen_snippet in chosen_snippets]
            #collected_snippets.setdefault(label, sample_snippets)
            sample_snippet = "\n\n".join([self.append_neighbors(chosen_snippet_id, neighbor_count) for chosen_snippet_id in chosen_snippets])
            collected_snippets.setdefault(label, sample_snippet)
        #print('parsed for clusters: ', len(collected_snippets))
        outline = await self.client.generate_outline(collected_snippets, name, description_mode)
        return outline

    def get_header_embeddings(self, doc_embeddings, doc_available, source_positions):
        """
        doc_embeddings: dict {номер_документа: embedding}
        doc_available: dict {id_ссылки: номер_документа}
        source_positions: dict {id_ссылки: ((header_level, header_text), paragraph_number)}
    
        Возвращает словарь вида {(header_level, header_text): embedding, ...} для заголовков h2, 
        где embedding – эмбеддинг первого доступного документа в группе, 
        охватывающей данный h2 и все подразделы до следующего h2.
        """
        header_to_embedding = {}
        current_header = None       
        current_links = []        
        
        for link_id, ((level, header_text), para_num) in source_positions.items():
            if int(level[1]) == 2 or int(level[1]) == 1:
                if current_header is not None and current_links:
                    emb = None
                    for lid in current_links:
                        if lid in doc_available:
                            doc_num = doc_available[lid]
                            if doc_num in doc_embeddings:
                                emb = doc_embeddings[doc_num]
                                break
                    if emb is not None:
                        header_key = (current_header[0], current_header[1])
                        if current_header not in header_to_embedding:
                            #print('Emb for head: ', current_header)
                            header_to_embedding[header_key] = emb

                current_header = (level, header_text)
                current_links = [link_id]
            else:
                if current_header is not None:
                    current_links.append(link_id)

        if current_header is not None and current_links:
            emb = None
            for lid in current_links:
                if lid in doc_available:
                    doc_num = doc_available[lid]
                    if doc_num in doc_embeddings:
                        emb = doc_embeddings[doc_num]
                        break
            if emb is not None:
                header_key = (current_header[0], current_header[1])
                header_to_embedding[header_key] = emb
        return header_to_embedding

    async def rank_outline(self, name, neighbor_count=0, forced_cluster_num=0, mode=0, page=None, description_mode=0):
        '''
        Оценка генерации плана статьи через кластеризацию источников
        '''
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        snippets_filtered = [sn[1] for sn in self.snippets.items() if sn[0][0] == name]
        #print('found snippets: ', len(snippets_filtered))
        snippets_id = [sn[0] for sn in self.snippets.items() if sn[0][0] == name]
        embeddings = self.get_embeddings(name)
        #print('emb len: ', len(embeddings))
        clusters_centers = None
        if mode:
            id_to_embedding = {}
            for i, snippet_id in enumerate(snippets_id):
                if snippet_id[1] not in id_to_embedding:
                    id_to_embedding[snippet_id[1]] = embeddings[i]
            #print('Count of emb ids: ', len(id_to_embedding))
            doc_num = {
                ref_link: doc_id for doc_id, ref_link in enumerate(page.downloaded_links)
            }
            #print('Downloaded docs: ', doc_num)
            header_to_embedding = self.get_header_embeddings(id_to_embedding, doc_num, page.references_positions)
            clusters_centers = list(header_to_embedding.values())
            #print(len(header_to_embedding))
            #print(list(header_to_embedding.keys())[0], ' ', len(list(header_to_embedding.values())[0]))
        outline = await self.k_means_method(name, snippets_filtered, snippets_id, embeddings, neighbor_count, forced_cluster_num, clusters_centers, description_mode)
        #print(outline)
        parsed_headings = []
        for line in outline.split('\n'):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                title = line[level:].strip()
                if title:
                    parsed_headings.append((level, title))
        heads = [head[1] for head in parsed_headings]
        pred_emb = self.model_embeddings.encode(heads, normalize_embeddings=True)
        true_heads = []
        for headings in page.filtered_outline.keys():
            true_heads.append(headings[1])
        ref_emb =  self.model_embeddings.encode(true_heads, normalize_embeddings=True)
        sims = pred_emb @ ref_emb.T
        precision_scores = sims.max(axis=1).mean()
        recall_scores = sims.max(axis=0).mean()
        if precision_scores + recall_scores == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision_scores * recall_scores / (precision_scores + recall_scores)
        
        return precision_scores, recall_scores, f1

    async def rank_section(self, page):
        section_to_sn = self.section_to_snippets(page)
        ranked_section = AsyncList()
        for section, snip in section_to_sn: