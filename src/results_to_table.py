import pandas as pd
import numpy as np
from collections import defaultdict


def ranking_results(data: list, article_names: list[str]):
    df = pd.DataFrame(data, columns=['NDCG', 'R-Precision'])
    df.index = article_names
    print(df)

def outline_results(data: list, article_names: list[str]):
    df = pd.DataFrame(data, columns=['Precision', 'Recall', 'F1-Score'])
    df.index = article_names
    print(df)

def section_results(data: list, article_names: list[str]):
    sections_data = []
    
    for article_idx, (article_metrics, article_name) in enumerate(zip(data, article_names)):
        num_sections = len(article_metrics[0])
        
        for section_idx in range(num_sections):
            sections_data.append({
                'article_id': article_idx + 1,
                'section_id': section_idx + 1,
                'topic': article_name,
                'precision': float(article_metrics[0][section_idx]),
                'recall': float(article_metrics[1][section_idx]),
                'f1': float(article_metrics[2][section_idx]),
                'rouge_l': float(article_metrics[3][section_idx]),
                'bleu': float(article_metrics[4][section_idx])
            })

    df_sections = pd.DataFrame(sections_data)
    
    df_articles = df_sections.groupby(['article_id', 'topic']).agg({
        'precision': 'mean', # просто среднее по метрикам
        'recall': 'mean',
        'f1': 'mean',
        'rouge_l': 'mean',
        'bleu': 'mean',
        'section_id': 'count'  # количество секций
    }).rename(columns={'section_id': 'num_sections'}).reset_index()

    print(df_articles.to_string(index=False))