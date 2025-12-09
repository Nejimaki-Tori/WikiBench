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


class WikiEvaluater():
    def __init__(self, device=None, encoder=None, evaluater=None):
        self.device = device
        self.encoder = encoder
        self.evaluater = evaluater
    
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

    def rank_query(self, ranking, true_name, only_rank_bm=False):
        if not only_rank_bm:
            ranking = sorted(ranking, reverse=True)
        relevances = [1 if name == true_name else 0 for _, name in ranking]
        ndcg_score = self.ndcg(relevances)
        r_pr_score = self.r_precision(relevances)
        return ndcg_score, r_pr_score

    def rank_outline(self, outline, page):
        parsed_headings = []
        for line in outline.split('\n'):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                title = line[level:].strip()
                if title:
                    parsed_headings.append((level, title))
            else:
                parsed_headings.append((-1, line))
        heads = [head[1] for head in parsed_headings]
        pred_emb = self.encoder.encode(heads, prompt='Classification', normalize_embeddings=True, device=self.device)
        true_heads = []
        for headings in page.filtered_outline.keys():
            true_heads.append(headings[1])
        ref_emb =  self.encoder.encode(true_heads, prompt='Classification', normalize_embeddings=True, device=self.device)
        sims = pred_emb @ ref_emb.T
        precision_scores = sims.max(axis=1).mean()
        recall_scores = sims.max(axis=0).mean()
        if precision_scores + recall_scores == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision_scores * recall_scores / (precision_scores + recall_scores)
        
        return precision_scores, recall_scores, f1

    def bertscore(self, gold_text, gen_text):
        ref_emb =  self.encoder.encode(
            [s.text for s in razdel.sentenize(gold_text)], 
            prompt='Classification', 
            normalize_embeddings=True, 
            device=self.device
        )

        pred_emb = self.encoder.encode(
            [s.text for s in razdel.sentenize(gen_text)], 
            prompt='Classification',
            normalize_embeddings=True, 
            device=self.device
        )

        sims = pred_emb @ ref_emb.T
        precision_scores = sims.max(axis=1).mean()
        recall_scores = sims.max(axis=0).mean()

        if precision_scores + recall_scores == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision_scores * recall_scores / (precision_scores + recall_scores)

        return precision_scores, recall_scores, f1

    def rouge_L(self, ref, pred):
        rouge = Rouge()
    
        score = rouge.get_scores(pred, ref)[0]["rouge-l"]["f"]
        return score
    
    def bleu_score(self, ref_text, pred_text):
        ref_tokens = [t.text for t in tokenize(ref_text)]
        pred_tokens = [t.text for t in tokenize(pred_text)]
        smooth = SmoothingFunction().method1
    
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        return score

    async def rank_sections(self, result, page, need_qa=False):
        pr_scores = []
        rec_scores = []
        f1_scores = []
        r_l_scores = []
        bleu_scores = []
        qa_sim_scores = []
        qa_cov_scores = []
        for section in page.filtered_outline.keys():
            gen_text = result.get(section)
            gold_text = page.filtered_outline[section]
            if not gen_text or gen_text == -1:
                continue

            precision_scores, recall_scores, f1 = self.bertscore(gold_text, gen_text)
            
            pr_scores.append(precision_scores)
            rec_scores.append(recall_scores)
            f1_scores.append(f1)

            r_l = self.rouge_L(gold_text, gen_text)
            r_l_scores.append(r_l)

            bleu_sc = self.bleu_score(gold_text, gen_text)
            bleu_scores.append(bleu_sc)
        
        return pr_scores, rec_scores, f1_scores

    def mean_value(self, data):
        ndcg = [e[0] for e in data]
        pr = [e[1] for e in data]
        return sum(ndcg) / len(ndcg) if ndcg else 0.0, sum(pr) / len(pr) if pr else 0.0
    
    def calc(self, data):
        data = np.array(data)
        data1 = (data,)
        bootstrap_ci = bootstrap(data1, np.mean, confidence_level=0.95, n_resamples=len(data)-1)
        
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
            pr_mean, pr_min, pr_max = self.calc([e[0] for el in data for e in el])
            rec_mean, rec_min, rec_max = self.calc([e[1] for el in data for e in el])
            f_mean, f_min, f_max = self.calc([e[2] for el in data for e in el])

        return pr_mean, pr_min, pr_max, rec_mean, rec_min, rec_max, f_mean, f_min, f_max