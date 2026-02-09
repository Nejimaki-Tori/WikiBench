import torch
from sentence_transformers import SentenceTransformer
from wiki_utils import WikiUtils
from scipy.stats import bootstrap
import math
import os
import razdel
import numpy as np
from openai_utils import AsyncList
from rouge import Rouge
from razdel import tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from wiki_extract import get_downloaded_page


class WikiEvaluater():
    def __init__(self, device=None, encoder=None):
        self.device = device
        self.encoder = encoder
        self.rouge = Rouge()
        self.bleu_smooth = SmoothingFunction().method1

    def calculate_prf_bertscore(self, pred_text: list[str], ref_text: list[str]):
        pred_emb =  self.encoder.encode(
            pred_text, 
            prompt='Classification', 
            normalize_embeddings=True, 
            device=self.device
        )
        
        ref_emb =  self.encoder.encode(
            ref_text, 
            prompt='Classification', 
            normalize_embeddings=True, 
            device=self.device
        )

        sims = pred_emb @ ref_emb.T
        precision_scores = sims.max(axis=1).mean()
        recall_scores = sims.max(axis=0).mean()

        f1 = (2 * precision_scores * recall_scores / (precision_scores + recall_scores)) if (precision_scores + recall_scores) > 0 else 0.0

        return precision_scores, recall_scores, f1
    
    def dcg(self, relevances: list[int], k: int):
        return sum(rel / math.log2(i+2) for i, rel in enumerate(relevances[:k]))
    
    def ndcg(self, relevances: list[int], k: int = None):
        if k is None:
            k = len(relevances)
        dcg_score = self.dcg(relevances, k)
        sorted_relevances = sorted(relevances, reverse=True)
        idcg = self.dcg(sorted_relevances, k)
        return (dcg_score / idcg) if idcg > 0 else 0.0
    
    def r_precision(self, relevances: list[int]):
        r = sum(relevances)
        return sum(relevances[:int(r)])/r if r else 0.0

    def mean_value(self, data):
        ndcg = [e[0] for e in data]
        pr = [e[1] for e in data]
        return sum(ndcg) / len(ndcg) if ndcg else 0.0, sum(pr) / len(pr) if pr else 0.0
        
    def rank_query(self, ranking, true_name):
        ranking = sorted(ranking, reverse=True, key=lambda x: x[0])
        relevances = [1 if name == true_name else 0 for _, name in ranking]
        ndcg_score = self.ndcg(relevances)
        r_pr_score = self.r_precision(relevances)
        return ndcg_score, r_pr_score


    # --- OUTLINE ---
    def parse_outline(self, outline):
        parsed_headings = []
        for line in outline.split('\n'):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                title = line[level:].strip()
                if title:
                    parsed_headings.append(title)
        return parsed_headings

    def rank_outline(self, outline, article_name):
        page = get_downloaded_page(article_name)
        pred_heads = self.parse_outline(outline)
        ref_heads = [h[1] for h in page.filtered_outline.keys()]
        return self.calculate_prf_bertscore(pred_heads, ref_heads)

    def rouge_L(self, ref, pred):
        return self.rouge.get_scores(pred, ref)[0]["rouge-l"]["f"]
    
    def bleu_score(self, ref_text, pred_text):
        ref_tokens = [t.text for t in tokenize(ref_text)]
        pred_tokens = [t.text for t in tokenize(pred_text)]
    
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.bleu_smooth)

    def rank_sections(self, result, article_name: str):
        pr_scores = []
        rec_scores = []
        f1_scores = []
        r_l_scores = []
        bleu_scores = []
        page = get_downloaded_page(article_name=article_name)
        for section in page.filtered_outline.keys():
            gen_text = result.get(section)
            gold_text = page.filtered_outline[section]
            if not gen_text or gen_text == -1:
                continue

            pred_sentences = [s.text for s in razdel.sentenize(gen_text)]
            gold_sentences = [s.text for s in razdel.sentenize(gold_text)]
            precision_scores, recall_scores, f1 = self.calculate_prf_bertscore(pred_sentences, gold_sentences)
            
            pr_scores.append(precision_scores)
            rec_scores.append(recall_scores)
            f1_scores.append(f1)

            r_l = self.rouge_L(gold_text, gen_text)
            r_l_scores.append(r_l)

            bleu_sc = self.bleu_score(gold_text, gen_text)
            bleu_scores.append(bleu_sc)

        out = {"precision": pr_scores, "recall": rec_scores, "f1": f1_scores, 'rouge_l': r_l_scores, 'bleu': bleu_scores}
        return out
    
    def calc(self, data):
        data = np.array(data)
        data1 = (data,)
        bootstrap_ci = bootstrap(
            data1, 
            np.mean, 
            confidence_level=0.95, 
            n_resamples=len(data)*100,
            random_state=42
        )
        dist = bootstrap_ci.bootstrap_distribution
        mean = np.quantile(dist, q=0.5)
        min = np.quantile(dist, q=0.025)
        max = np.quantile(dist, q=0.975)
        return mean, min, max

    def bootstrap(self, data, is_flat=True):
        if is_flat: # for outline
            pr_mean, pr_min, pr_max = self.calc([e[0] for e in data])
            rec_mean, rec_min, rec_max = self.calc([e[1] for e in data])
            f_mean, f_min, f_max = self.calc([e[2] for e in data])
        else: # for sections
            pr_mean, pr_min, pr_max = self.calc([e for el in data for e in el[0]])
            rec_mean, rec_min, rec_max = self.calc([e for el in data for e in el[1]])
            f_mean, f_min, f_max = self.calc([e for el in data for e in el[2]])
            r_mean, r_min, r_max = self.calc([e for el in data for e in el[2]])
            bf_mean, b_min, b_max = self.calc([e for el in data for e in el[2]])
        if is_flat:
            return pr_mean, pr_min, pr_max, rec_mean, rec_min, rec_max, f_mean, f_min, f_max
        else:
            return pr_mean, pr_min, pr_max, rec_mean, rec_min, rec_max, f_mean, f_min, f_max, r_mean, r_min, r_max, bf_mean, b_min, b_max
