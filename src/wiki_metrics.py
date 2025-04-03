import os
import glob
import Stemmer
import nltk
from nltk.corpus import stopwords
from datasets import Dataset

def load_texts_from_directory(root_dir: str, file_extension="*.txt"):
    """
    Рекурсивно собираем все файлы с указанным расширением из root_dir.
    Возвращает список (article_path, text_content).
    """
    texts = []
    pattern = os.path.join(root_dir, "**", file_extension)
    for filepath in glob.iglob(pattern, recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            texts.append((filepath, content))
    return texts

def chunk_text(text: str, window_size=2000, overlap=512):
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

def is_cyrillic(word):
    return any(c.lower() not in "abcdefghijklmnopqrstuvwxyz" for c in word)

ru_stopwords = stopwords.words("russian")
en_stopwords = stopwords.words("english")
combined_stopwords = set(ru_stopwords + en_stopwords)

russian_stemmer = Stemmer.Stemmer("russian")
english_stemmer = Stemmer.Stemmer("english")

def ultra_stemmer(words_list):
    return [
        russian_stemmer.stemWord(word) if is_cyrillic(word) else english_stemmer.stemWord(word)
        for word in words_list
    ]

def tokenize_corpus(root_dir: str):
    all_texts = load_texts_from_directory(root_dir, file_extension="*.txt")
    snippets = []
    for file_path, content in all_texts:
        file_snippets = chunk_text(content, window_size=2000, overlap=512)
        for idx, snippet in enumerate(file_snippets):
            snippets.append({
                "doc_id": file_path, 
                "snippet_index": idx, 
                "text": snippet
            })
    
    corpus_texts = [snip["text"] for snip in snippets]
    corpus_tokens = bm25s.tokenize(
        corpus_texts, 
        stopwords=combined_stopwords, 
        stemmer=ultra_stemmer
    )

    output_dir = r"Generation\Utils\bm25_index"
    retriever.save(output_dir, corpus=corpus_texts)

    dataset = Dataset.from_dict({
        "doc_id": [s["doc_id"] for s in snippets],
        "snippet_index": [s["snippet_index"] for s in snippets],
        "text": [s["text"] for s in snippets]
    })
    dataset_dir = r"Generation\Utils\text_corpus"
    dataset.save_to_disk(dataset_dir)
    
    return corpus_tokens, snippets

def tokenize_query(query):
    return bm25s.tokenize(query, stopwords=combined_stopwords, stemmer=ultra_stemmer)