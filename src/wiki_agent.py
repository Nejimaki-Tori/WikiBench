# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import random
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
from openai_utils import AsyncList


class WikiAgent(WikiUtils):
    def __init__(self, client, device=None, encoder=None, pre_load=False):
        super().__init__(device=device, encoder=encoder, pre_load=pre_load)
        self.client = client
    
    async def create_ranking(self, query: str, top_k: int, true_name: str):
        '''
        Ранжирование поисковой выдачи с помощью LLM
        '''
        tokenized_query = self.tokenize_query(query)
        results, scores = self.retriever.retrieve(tokenized_query, k=top_k, show_progress=False,)
        source_names = []
        snippets_list = list(self.snippets.items())
        chosen_snippets = [results[0, i] for i in range(results.shape[1])]
        source_texts = [snippets_list[idx][1] for idx in chosen_snippets]
        source_names = [snippets_list[idx][0][0] for idx in chosen_snippets] # другой вид сохранения сниппетов! вроде поправил
        probs = await self.client.filter_sources(true_name, source_texts)
        ranking = [(prob[1], name) for name, prob in zip(source_names, probs)]
        #print(source_texts)
        #print(ranking)
        return ranking

    def append_neighbors(self, chosen_snippet, neighbor_count=2):
        '''
        Присоединение соседних сниппетов к конкретному сниппету
        Количество соседей указывает на число добавляемых сниппетов слева и справа соответственно
        '''
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

    async def k_means_method(self, name, snippets_filtered, snippets_id, embeddings, neighbor_count, forced_cluster_num, clusters_centers, description_mode=0):
        '''
        Кластеризация документов для составления плана статьи
        '''
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
            sample_snippet = "\n\n".join([self.append_neighbors(chosen_snippet_id, neighbor_count) for chosen_snippet_id in chosen_snippets])
            collected_snippets.setdefault(label, sample_snippet)
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
        
        for link_id, pos_lst in source_positions.items():
            (level, header_text), para_num = pos_lst[0]
            #print(level, header_text[:5], para_num)
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

    async def create_outline(self, name, neighbor_count=0, forced_cluster_num=0, mode=0, page=None, description_mode=0):
        '''
        Генерация плана к статье посредством кластеризации документов и выделении в них ключевых тем
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
                ref_link: doc_id + 1 for doc_id, ref_link in enumerate(page.downloaded_links)
            }
            #print('Downloaded docs: ', doc_num)
            header_to_embedding = self.get_header_embeddings(id_to_embedding, doc_num, page.references_positions)
            clusters_centers = list(header_to_embedding.values())
            #print('Found hints: ', len(header_to_embedding))
            #return
            #print(list(header_to_embedding.keys())[0], ' ', len(list(header_to_embedding.values())[0]))
        outline = await self.k_means_method(name, snippets_filtered, snippets_id, embeddings, neighbor_count, forced_cluster_num, clusters_centers, description_mode)
        return outline

    def group_snippets(self, selected_emb, texts_final):
        '''
        Группировка похожих текстов для исключения дублирования информации
        '''
        sim_matrix = selected_emb @ selected_emb.T
        mask = sim_matrix >= 0.8
        np.fill_diagonal(mask, 0)
        graph = csr_matrix(mask)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(texts_final[idx])
        return list(groups.values())
        
    def filter_snippets(self, section, snips, page):
        '''
        Отбор сниппетов наиболее подходящих для написания текста секции
        '''
        if not page.filtered_outline[section]:
            return None

        # Отбор только тех сниппетов, что имеют большое сходство с текстом секции в статье
        texts_only = [elem[1] for elem in snips] # тексты сниппетов к конкретной секции
        emb_snips = self.encoder.encode(texts_only, normalize_embeddings=True, device=self.device)
        q_emb = self.encoder.encode(page.filtered_outline[section], normalize_embeddings=True, device=self.device)
        cosine_scores = emb_snips @ q_emb.T
        new_ranked = [(sn, cos, emb) for sn, cos, emb in zip(snips, cosine_scores, emb_snips) if cos >= 0.5]
        filtered_final = sorted(new_ranked, key=lambda x: (x[0][0][1], x[0][0][2])) # сортировка сниппетов по порядку их встречаемости в тексте 
        selected_emb = np.array([elem[2] for elem in filtered_final])
        texts_final = [elem[0][1] for elem in filtered_final]
        
        if len(texts_final) <= 1:
            return [texts_final] if len(texts_final) == 1 else None

        groups = self.group_snippets(selected_emb, texts_final) # группировка

        return groups

    async def get_section(self, section, snips, page):
        filtered_snippets = self.filter_snippets(section, snips, page)
        if not filtered_snippets:
            return -1
        #print('Generating groups')
        summarized_snippets = await self.client.summarize_groups(filtered_snippets)
        #print('Number of grouped snippets: ', len(summarized_snippets))
        #print('Generation section')
        generated_text = await self.client.generate_section(section, summarized_snippets, page.name)
        #print('Generated for section (hr): ', section)
        return generated_text

    async def create_sections(self, page):
        section_to_sn = self.section_to_snippets(page)
        sections = AsyncList()
        secs = []
        for section, snips_id in section_to_sn.items():
            #print('Generating section: ', section)
            sections.append(self.get_section(section, snips_id, page))
            secs.append(section)
            #break
        await sections.complete_couroutines(batch_size=4)
        sections = await sections.to_list()
        result = {}
        for sn, txt in zip(secs, sections):
            result[sn] = txt
        return result