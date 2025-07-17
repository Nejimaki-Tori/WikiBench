import torch
from sentence_transformers import SentenceTransformer
from wiki_utils import WikiUtils
from scipy.stats import bootstrap
import math

class WikiEvaluater():
    def __init__(self, device, encoder):
        self.device = device
        self.encoder = encoder
    
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

    def rank_query(self, ranking, true_name):
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

    def rank_sections(self, result, page, model_name="tmp"):
        pr_scores = []
        rec_scores = []
        f1_scores = []
        model_name = re.sub(r'[<>:"/\\|?*]', '', model_name)
        dir_name = os.path.join("Gen_articles", model_name)
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, page.cleared_name + '.txt')
        with open(path, 'w', encoding='utf-8') as file:
            for section in page.filtered_outline.keys():
                if ((section not in result) or result[section] == -1) and section[0] == 'h2':
                    print(section, file=file)
                    print(f"BERTScore Precision: {0:.4f}", file=file)
                    print(f"BERTScore Recall:    {0:.4f}", file=file)
                    print(f"BERTScore F1:        {0:.4f}", file=file)
                    print(file=file)
                    print(file=file)
                    continue
                elif section not in result or result[section] == -1:
                    continue
                ref_emb =  self.encoder.encode(
                    [s.text for s in razdel.sentenize(page.filtered_outline[section])], 
                    prompt='Classification', 
                    normalize_embeddings=True, 
                    device=self.device
                )
                pred_emb = self.encoder.encode(
                    [s.text for s in razdel.sentenize(result[section])], 
                    prompt='Classification',
                    normalize_embeddings=True, 
                    device=self.device
                )
                sims = pred_emb @ ref_emb.T
                precision_scores = sims.max(axis=1).mean()
                recall_scores = sims.max(axis=0).mean()
                pr_scores.append(precision_scores)
                rec_scores.append(recall_scores)
                if precision_scores + recall_scores == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision_scores * recall_scores / (precision_scores + recall_scores)
                f1_scores.append(f1)
                print(section, file=file)
                print(f"BERTScore Precision: {precision_scores:.4f}", file=file)
                print(f"BERTScore Recall:    {recall_scores:.4f}", file=file)
                print(f"BERTScore F1:        {f1:.4f}", file=file)
                print(file=file)
                print(result[section], file=file)
                print(file=file)
                print(file=file)
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
        return mean

    def bootstrap(self, data, is_flat=True):
        if is_flat: # for outline
            pr = self.calc([e[0] for e in data])
            rec = self.calc([e[1] for e in data])
            f = self.calc([e[2] for e in data])
        else: # for sections
            pr = calc([e[0] for el in data for e in el])
            rec = calc([e[1] for el in data for e in el])
            f = calc([e[2] for el in data for e in el])

        return pr, rec, f