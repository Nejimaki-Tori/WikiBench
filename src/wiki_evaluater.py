import torch
from sentence_transformers import SentenceTransformer
from wiki_utils import WikiUtils
from scipy.stats import bootstrap
import math
import os
import razdel
import numpy as np
from openai_utils import AsyncList

QUESTIONS_COVERAGE_PROMPT = """Ты — эксперт в оценивании качества секций статьи. Твоя задача — точно определить, насколько предоставленный фрагмент текста секции отвечает на конкретный вопрос о ключевых аспектах этой секции.

Вопрос:
{question}
Текст секции статьи:
{text}

Содержится ли в этом тексте ответ на вопрос?
Начни ответ с {yes}, если ответ есть, или с {no}, если ответа нет.
"""

GOLD_QUESTIONS_PROMPT = """На основе содержания данной секции статьи сформируй несколько ключевых вопросов, ответы на которые можно однозначно дать, прочитав только эту секцию.

Секция статьи:
---
{ref_section}
---
Каждый вопрос писать с новой строки, без лишних комментариев.
"""

ANSWER_PROMPT = """На основе содержания данной секции статьи сформируй ответ на заданный ключевой вопрос. Отвечай **строго на основании текста секции**, не добавляя ничего лишнего.

Секция статьи:
{ref_section}

Ключевой вопрос:
{key_question}

---
Пиши только ответ, без повторения вопроса.
"""

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

        def similarity(self, a, b):
            emb_1 = self.encoder.encode(a, device=self.device)
            emb_2 = self.encoder.encode(b, device=self.device)
    
            return round(float(self.encoder.similarity(emb_1, emb_2).item()), 3)


    async def compute_coverage(
            self,
            questions,
            summary,
            positive_choice="YES",
            negative_choice="NO"
    ):
        probs = AsyncList()
    
        for q in questions:
            myprompt = QUESTIONS_COVERAGE_PROMPT.format(question=q, text=summary, yes=positive_choice, no=negative_choice)
            probs.append(self.evaluater.get_probability(myprompt, rep_penalty=1.0, max_tokens=10))
    
        await probs.complete_couroutines(batch_size=40)
        results = await probs.to_list()

        flags = []
    
        for res in results:
            probs = {positive_choice: [], negative_choice: []}

            for token_info in response.logprobs.content:
                for variant in token_info.top_logprobs:
                    key = variant.token.strip()
                    if key == positive_choice or key == negative_choice:
                        probs[key].append(math.exp(variant.logprob))
    
            prob_pos = max(probs[positive_choice], default=0.0)
            prob_neg = max(probs[negative_choice], default=0.0)
    
            if prob_neg > prob_pos:
                prob_val = 1 - prob_neg
            else:
                prob_val = prob_pos

            flags.append(1 if prob_val >= 0.75 else 0)
    
        coverage = sum(flags) / len(flags) if flags else 0.0
    
        return coverage, flags
    
    
    async def generate_key_questions(self, ref_section):
        myprompt = GOLD_QUESTIONS_PROMPT.format(ref_section=ref_section)
        
        res = await self.evaluater.get_completion(myprompt, max_tokens=512)
        result = extract_response(res)

        questions = [q for q in result.split('\n') if q.strip()]
        
        return questions
    
    
    async def get_answer(self, ref_section, key_question):
        myprompt = ANSWER_PROMPT.format(ref_section=ref_section, key_q=key_question)
        res = await self.client_eval.get_completion(myprompt, max_tokens=512)

        answer = extract_response(res)
    
        return answer

    async def generate_answers(self, ref_annotation, questions, cov_flags=None):
        answers = AsyncList()

        if cov_flags:
            for question, flag in zip(questions, cov_flags):
                if flag == 0:
                    answers.append('')
                else:
                    answers.append(self.get_answer(ref_annotation, question))
        else:
            for question in questions:
                answers.append(self.get_answer(ref_annotation, question))

        await answers.complete_couroutines(batch_size=40)
        answers = await answers.to_list()

        return answers
    
    
    def compute_answer_similarity(self, questions, cov_flags, answers_gold, answers_gen):
        sims = []
    
        for flag, gen, gold in zip(cov_flags, answers_gen, answers_gold):
            if flag == 0:
                sims.append(0.0)
            else:
                sims.append(self.similarity(gen, gold))
    
        return sum(sims) / len(sims) if sims else 0.0

    async def compute_similarity(self, ref_section, gen_section, q, a):
        if not q or not a:
            print('wrong!')
            questions = await self.generate_key_questions(ref_section)
            #print(questions)
            #print()
            #print()
            answers_gold = await self.generate_answers(ref_section, questions, None)
            #print(answers_gold)
            #print()
            #print()
        else:
            questions = q
            answers_gold = a
        coverage, cov_flags = await self.compute_coverage(questions, gen_annotation)
        #print(cov_flags)
        answers_gen = await self.generate_answers(gen_section, questions, cov_flags)
        #print(answers_gen)
        #print()
        #print()
        answer_similarity = self.compute_answer_similarity(questions, cov_flags, answers_gold, answers_gen)

        return coverage, answer_similarity

    async def rank_sections(self, result, page):
        pr_scores = []
        rec_scores = []
        f1_scores = []
        qa_sim_scores = []
        qa_cov_scores = []
        for section in page.filtered_outline.keys():
            gen_text = result.get(section)
            gold_text = page.filtered_outline[section]
            if not gen_text or gen_text == -1:
                pr_scores.append(0)
                rec_scores.append(0)
                f1_scores.append(0)
                qa_sim_scores.append(0)
                qa_cov_scores.append(0)
                continue
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
            pr_scores.append(precision_scores)
            rec_scores.append(recall_scores)
            if precision_scores + recall_scores == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision_scores * recall_scores / (precision_scores + recall_scores)
            f1_scores.append(f1)
            
            qa_cov, qa_sim = await self.compute_similarity(gold_text, gen_text)
            qa_cov_scores.append(qa_cov)
            qa_sim_scores.append(qa_sim)
                
        return pr_scores, rec_scores, f1_scores, qa_cov_scores, qa_sim_scores

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