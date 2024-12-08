import argparse
import math
import os
import re
from collections import defaultdict, Counter
import time
from functools import lru_cache
from files import porter


# Extract documents and divides them into terms.
def extract_documents(directory):
    documents = {}
    for entry in os.scandir(directory):
        with open(entry.path, "r") as file:
            doc = file.read()
            # convert to lower case
            doc = doc.lower()
            # Divide the document into terms using regular expression
            # this only allows the lower case letters and numbers
            terms_data = re.findall(r"[a-z']+|\d+(?:\.\d+)?", doc)
            doc_id = entry.name
            documents[doc_id] = terms_data
    return documents


# Remove stopwords
def remove_stopwords(terms, stopwords):
    # Create a new list contains only the terms not present in the stopwords list
    return [term for term in terms if term.lower() not in stopwords]


# Load the stopwords from the file
def load_stopwords(stopwords_file):
    with open(stopwords_file, "r") as file:
        stopwords = set(line.strip().lower() for line in file)
    return stopwords


stemmer = porter.PorterStemmer()  # Initialize the stemmer


# Use lru_cache to improve the speed and efficiency.
@lru_cache(maxsize=128000)  # Set max size to limit the number of cached items
def stem_word(word):
    return stemmer.stem(word)  # Return the stemmed word


# perform the stem operation
def stemming(terms):
    return [stem_word(word) for word in terms]


def calc_BM25(term, documents, N, df, average_doc_length, k1, b, documents_lengths):  # 计算BM25
    # Calculate idf
    idf = math.log((N - df + 0.5) / (df + 0.5), 2)
    # Create a dict to store the calculated BM25
    BM25 = {}
    for doc_id, tf in documents[term].items():
        doc_length = documents_lengths[doc_id]

        if tf != 0:
            # calculate
            weight = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / average_doc_length))))
            # Store the bm25
            BM25[doc_id] = weight
    return BM25


# Create the index and store the BM25 value into it
def create_index(documents_directory, index_file):
    # Extract and preprocess
    documents = extract_documents(documents_directory)
    # process the terms
    for document_id, terms in documents.items():
        # Remove stopwords
        terms_without_stopwords = remove_stopwords(terms, stopwords)
        # Perform stemming
        terms_after_stem = stemming(terms_without_stopwords)
        documents[document_id] = Counter(terms_after_stem)
    # Create an index dictionary and calculate the total number of docs
    index = defaultdict(dict)
    N = len(documents)

    # Calculate the doc lengths
    documents_lengths = {}
    total_doc_length = 0

    for document_id, terms in documents.items():
        doc_length = sum(terms.values())
        documents_lengths[document_id] = doc_length
        total_doc_length += doc_length

        for term, tf in terms.items():
            index[term][document_id] = tf
    # Calculate the average document length
    average_doc_length = total_doc_length / N

    # calculate document frequency
    doc_freq = {}
    for terms in documents.values():
        for term in terms:
            if term in doc_freq:
                doc_freq[term] += 1
            else:
                doc_freq[term] = 1

    # Calculate BM25 and create the index
    for term, doc_f in doc_freq.items():
        weights = calc_BM25(term, index, N, doc_f, average_doc_length, k1, b, documents_lengths)
        index[term] = weights

    # Write the index to the file
    with open(index_file, "w") as file:
        for term, weights in index.items():
            file.write(f"{term}: ")
            file.write(" ".join(f"{document_id}:{weight}" for document_id, weight in weights.items()))
            file.write("\n")

    return index

# load the index file, if there is an index.txt file, load it directly without further operation
def load_index(index_file):
    index = {}
    # Open the index file and process it
    with open(index_file, "r") as file:
        for line in file:
            term, weights = line.strip().split(": ")
            weights = weights.split()
            # save the data into index
            index[term] = {doc_id: float(weight) for doc_id, weight in (item.split(":") for item in weights)}
    return index


# Process the query
def parse_query(query):
    # Split the query into individual terms
    query_terms = query.lower().split()
    # Preprocess the terms
    newTerms_without_stopwords = remove_stopwords(query_terms, stopwords)
    newTerms_after_stem = stemming(newTerms_without_stopwords)

    return newTerms_after_stem


# retrieve docs and sort
def retrieve_documents(query, index):
    query_weights = {}
    max_weight = 0  # 最大权重值

    for term in query:
        if term in index:
            for document_id, weight in index[term].items():
                query_weights[document_id] = query_weights.get(document_id, 0) + weight
                max_weight = max(max_weight, weight)
    # Sort the search results in descending order
    return sorted(query_weights.items(), key=lambda x: x[1], reverse=True)


# Interactive mode
def interactive_mode(index):
    while True:
        query = input("Please enter a query or 'QUIT': ")
        # User quit
        if query == "QUIT":
            break

        # Process the query
        query_terms = parse_query(query)
        results = retrieve_documents(query_terms, index)

        print("Results for query"+'('+query+')')
        # Check if there are any results
        if len(results) == 0:
            print("No results found. Please try another query like 'library information conference'.")
        else:
            # Print the rank, document id, and similarity
            for rank, (document_id, similarity_score) in enumerate(results[:15], start=1):
                print(f"{rank:<6}{document_id:<13}{similarity_score:.4f}")
        print()


# Automatic mode
def automatic_mode(index, queries_file, results_file):
    with open(queries_file, "r") as file:
        queries = file.readlines()

    with open(results_file, "w") as file:
        for query in queries:  # Process the query
            query_id, query_text = query.strip().split(" ", 1)
            query_terms = parse_query(query_text)
            results = retrieve_documents(query_terms, index)

            for rank, (document_id, similarity_score) in enumerate(results, start=1):
                if similarity_score >= 0:
                    # Print the rank, document id, and similarity
                    file.write(f"{query_id} {rank} {document_id} {similarity_score:.4f}\n")

k1 = 1
b = 0.75

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode")
    args = parser.parse_args()

    # file paths
    stopwords_file = "files/stopwords.txt"
    documents_directory = "./documents"
    index_file = "./index.txt"
    queries_file = "files/queries.txt"
    results_file = "./results.txt"
    # only load once stop words
    stopwords = load_stopwords(stopwords_file)

    # check whether the index file exist or not
    if os.path.exists(index_file):
        start_time1 = time.process_time()

        # load index
        index = load_index(index_file)
        end_time1 = time.process_time()
        print("Index loaded successfully.")
        print(f"Load index cost: {end_time1 - start_time1} s")
    else:
        start_time1 = time.process_time()
        # The index file does not exist. You need to create an index
        index = create_index(documents_directory, index_file)
        end_time1 = time.process_time()
        print("Index created and saved.")
        print(f"Index cost: {end_time1 - start_time1} s")

    # check the mode
    if args.mode == "interactive":
        # interactive mode
        interactive_mode(index)
    elif args.mode == "automatic":
        # Calculate the time
        start_time = time.process_time()
        # automatic mode
        automatic_mode(index, queries_file, results_file)
        end_time = time.process_time()
        print(f"Query cost: {end_time - start_time} s")

