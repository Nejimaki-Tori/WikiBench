# -*- coding: utf-8 -*-

import re
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wiki_utils import SnippetKey
from openai_utils import AsyncList


class WikiAgent:
    def __init__(
        self, 
        client,
        utils,
        device=None, 
        encoder=None
    ):  
        self.client = client
        self.utils = utils
        self.device = device or self.utils.device
        self.encoder = encoder or self.utils.device

        self.snippet_keys = None
        self.snippet_texts = None
        self.snippets_by_article_grouped = None
        self.is_cache_built = False

    def build_cache(self):
        if self.is_cache_built:
            return

        if self.utils.snippets is None:
            raise ValueError('No snippets!')

        items = list(self.utils.snippets.items())
        self.snippet_keys = [k for k, _ in items]
        self.snippet_texts = [v for _, v in items]

        grouped: dict[tuple[str, int], list[tuple[int, str]]] = {} # groups for snippets per article
        for snippet_key, snippet_text in items:
            grouped.setdefault((snippet_key.article_name, snippet_key.source_id), []).append((snippet_key.snippet_id, snippet_text))

        for group_key in grouped:
            grouped[group_key].sort(key=lambda x: x[0])

        self.snippets_by_article_grouped = grouped
        self.is_cache_built = True
    
    async def create_ranking(
        self, 
        query: str, 
        top_k: int, 
        true_article_name: str
    ):
        '''
        Ranking search query with LLM
        '''
        if self.utils.retriever is None:
            raise ValueError('BM25 has not been created!')

        if self.utils.snippets is None:
            raise ValueError('No snippets!')

        self.build_cache()
        
        tokenized_query = self.utils.tokenize_query(query)
        results, scores = self.utils.bm25.retrieve(
            tokenized_query, 
            k=top_k, 
            show_progress=False
        )
        indices = result[0].tolist()
        source_texts = [self.snippet_texts[i] for i in indices]
        source_names = [self.snippet_keys[i].article_name for i in indices]
        
        probs = await self.client.filter_sources(true_article_name, source_texts)
        ranking = [(probability, article_name) for article_name, (idx, probability) in zip(source_names, probs)]
        return ranking

    def append_neighbors(self, chosen_snippet: SnippetKey, neighbor_count: int = 1):
        '''
        Appending neighboring snippets to given snippet.
        Neighbor count shows how much snippets to add on left and right side.
        '''
        if self.utils.snippets is None:
            raise ValueError('No snippets!')
            
        self.build_cache()

        group_key = (chosen_snippet.article_name, chosen_snippet.source_id)
        grouped = self.snippets_by_article_grouped(group_key)

        target_id = chosen_snippet.snippet_id
        start = target_id - neighbor_count
        end = target_id + neighbor_count

        texts = [text for snippet_id, text in grouped if start <= snippet_id <= end]

        return " ".join(texts)

    async def k_means_method(
        self,
        snippets_id: list[SnippetKey], 
        embeddings: np.ndarray, 
        neighbor_count: int, 
        forced_cluster_num: int = 0, 
        clusters_centers: np.ndarray, 
        article_name: str, 
        description_mode: int = 0
    ):
        '''
        Documents clusterization for constructing an article outline
        '''

        snippet_to_embedding = {snippet_id: embedding for snippet_id, embedding in zip(snippets_id, embeddings)}

        # chosing number of clusters
        if forced_cluster_num:
            k = forced_cluster_num
        elif cluster_centers:
            k = len(clusters_centers)
        else:
            k = min(10, max(2, len(embeddings) // 10 or 2))

        # chosing KMeans configuration
        if clusters_centers:
            init = np.vstack(clusters_centers)
            kmeans = KMeans(n_clusters=k, init=init, n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)

        # obtaining cluster labels
        labels = kmeans.fit_predict(embeddings)

        # constructing clusters
        clusters = {}
        for label, snippet_id in zip(labels, snippets_id):
            clusters.setdefault(label, []).append((snippet_id, snippet_to_embedding[snippet_id]))

        collected_snippets = {}
        for label, snippet_id_and_emb in clusters.items(): # for each cluster
            cluster_emb = np.array([elem for _, elem in snippet_id_and_emb]) # collect its embeddings
            if len(cluster_emb) <= 5: # return early if it does not have more then five snippets
                chosen_snippets = [sn_id for sn_id, _ in snippet_id_and_emb]
            else:
                cluster_center = cluster_emb.mean(axis=0) # cluster center
                distances = np.linalg.norm(cluster_emb - cluster_center, axis=1) # distance to cluster center
                top_idx = np.argsort(distances)[:5] # top 5 closest
                chosen_snippets = [snippet_id_and_emb[i][0] for i in top_idx] # obtaining snippet ids of closest elements

            # concatenating snippets in one text
            sample_snippet = '\n\n'.join(
                self.append_neighbors(chosen_snippet_id, neighbor_count) 
                for chosen_snippet_id in chosen_snippets
            )
            collected_snippets[label] = sample_snippet

        # generating outline
        outline = await self.client.generate_outline(collected_snippets, article_name, description_mode)
        return outline

    def get_header_embeddings(
        self, 
        doc_embeddings: dict[int, np.ndarray], 
        doc_available: dict[str, int], 
        source_positions: dict[str, list[tuple[tuple[str, str], int]]]
    ):
        '''
        doc_embeddings: dict {document_num: embedding}
        doc_available: dict {ref_id: document_num}
        source_positions: dict {ref_id: ((header_level, header_text), paragraph_number)}
    
        Returns dictionary {(header_level, header_text): embedding, ...} for h2 headings, 
        where embedding is an embedding for first available doc in the group for this h2
        '''
        
        header_to_embedding = {}
        current_header = None       
        current_links = []        
        
        for link_id, pos_lst in source_positions.items():
            (level, header_text), para_num = pos_lst[0]
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
        
    async def create_outline(
        self,  
        neighbor_count: int = 0, 
        forced_cluster_num=0, 
        mode: bool = False, 
        page=None, 
        description_mode=0,
        article_name: str = None
    ):
        '''
        Outline generation using clusterization
        '''
        if self.utils.snippets is None:
            raise ValueError('No snippets!')

        safe_name = page.cleared_name if page else article_name

        snippets_id = [
            snippet_key
            for snippet_key in self.utils.snippets.keys()
            if snippet_key.article_name == safe_name
        ]
        
        embeddings = self.get_embeddings(safe_name)

        clusters_centers = None
        if mode and page is not None:
            id_to_embedding = {}
            for i, snippet_key in enumerate(snippets_id):
                if snippet_id.source_id not in id_to_embedding:
                    id_to_embedding[snippet_id.source_id] = embeddings[i]

            doc_num = {
                ref_link: doc_id + 1
                for doc_id, ref_link in enumerate(page.downloaded_links)
            }

            header_to_embedding = self.get_header_embeddings(
                id_to_embedding, 
                doc_num, 
                page.references_positions
            )
            clusters_centers = list(header_to_embedding.values())

        outline = await self.k_means_method(
            name, 
            snippets_id, 
            embeddings, 
            neighbor_count, 
            forced_cluster_num, 
            clusters_centers, 
            description_mode
        )
        return outline

    def group_snippets(self, selected_emb, texts_final):
        '''
        Группировка похожих текстов для исключения дублирования информации
        '''
        sim_matrix = selected_emb @ selected_emb.T
        mask = sim_matrix >= 0.8
        np.fill_diagonal(mask, 0)
        
        graph = csr_matrix(mask)
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(texts_final[idx])
            
        return list(groups.values())
        
    def filter_snippets(self, section: tuple[str, str], snippets: list[tuple[SnippetKey, str]], page):
        '''
        Selecting most relevant snippets to the given section
        '''
        if not page.filtered_outline.get(section):
            return None

        if self.encoder is None:
            raise ValueError('No encoder!')

        # Only snippets that are similar to section text will be used
        texts_only = [snippet_text for _, snippet_text in snippets]
        
        emb_snippets = self.encoder.encode(
            texts_only, 
            normalize_embeddings=True, 
            device=self.device
        )
        q_emb = self.encoder.encode(
            page.filtered_outline[section], 
            normalize_embeddings=True, 
            device=self.device
        )
        cosine_scores = (emb_snippets @ q_emb.T).ravel()
        new_ranked = [
            (snippet_info, emb) 
            for snippet_info, cos, emb in zip(snippets, cosine_scores, emb_snippets) 
            if cos >= 0.6
        ]

        if not new_ranked:
            return None

        def sort_key(elem):
            key = elem[0][0] # SnippetKey
            return key.source_id, key.snippet_id

        filtered_final = sorted(new_ranked, key=sort_key) # sorting snippets in order they were in the text
        selected_emb = np.array([emb for _, emb in filtered_final])
        texts_final = [snippet_text for (_, snippet_text), _ in filtered_final]
        
        if len(texts_final) == 0:
            return None
            
        if len(texts_final) == 1:
            return [texts_final]

        groups = self.group_snippets(selected_emb, texts_final) # grouping

        return groups

    async def get_section(self, section, snippets: list[tuple[SnippetKey, str]], page):
        filtered_snippets = self.filter_snippets(section, snippets, page)
        if not filtered_snippets:
            return -1

        summarized_snippets = await self.client.summarize_groups(filtered_snippets)
        generated_text = await self.client.generate_section(section, summarized_snippets, page.name)

        return generated_text

    async def create_sections(self, page):
        section_to_sn = self.utils.section_to_snippets(page)
        
        sections = AsyncList()
        secstions_order = []
        
        for section, snips_id in section_to_sn.items():
            sections.append(self.get_section(section, snips_id, page))
            secstions_order.append(section)

        await sections.complete_couroutines(batch_size=4)
        results = sections.to_list()
        return dict(zip(sections_order, results))