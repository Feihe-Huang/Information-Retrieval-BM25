import os


# Calculate the precision
def calc_precision(results, rel_doc):
    # Calculate retrieved documents number
    ret_num = len(results)
    # If no documents are retrieved, return 0 precision
    if ret_num == 0:
        return 0.0
    # Count the number of both retrieved and relevant docs
    ret_rel_num = 0
    for doc_id, _ in results:
        if doc_id in rel_doc:
            ret_rel_num += 1

    precision = ret_rel_num / ret_num
    return precision


# calculate the recall
def calc_recall(results, rel_doc):
    # Calculate relevant documents number
    rel_num = len(rel_doc)
    # If no relevant documents, return 0 recall
    if rel_num == 0:
        return 0
    # Count the number of both retrieved and relevant docs
    ret_rel_num = 0
    for doc_id, _ in results:
        if doc_id in rel_doc:
            ret_rel_num += 1

    recall = ret_rel_num / rel_num
    return recall


# Calculate the p@10 value
def calc_p_at_10(results, rel_doc):
    # Calculate retrieved documents number, capped at 10
    ret_num = min(len(results), 10)
    # If no documents are retrieved, return 0
    if ret_num == 0:
        return 0
    # Count the number of both retrieved and relevant docs in the top 10 results
    ret_rel_num = 0
    for rank, (doc_id, _) in enumerate(results[:10], start=1):
        if doc_id in rel_doc:
            ret_rel_num += 1

    p_at_10 = ret_rel_num / ret_num
    return p_at_10


# Calculate the R precision value
def calc_r_precision(results, rel_doc, num_rel_doc):
    # Calculate the minimum number of retrieved doc and relevant documents
    k = min(len(results), num_rel_doc)

    if k == 0:
        return 0.0
    # Count the number of both retrieved and relevant docs in the top k results
    ret_rel_num = 0
    for rank, (document_id, _) in enumerate(results[:k], start=1):
        if document_id in rel_doc:
            ret_rel_num += 1
    # Calculate the R precision
    r_precision = ret_rel_num / num_rel_doc
    return r_precision


# calculate the MAP
def calc_average_precision(results, rel_doc):
    retrieved_num = len(results)
    rel_num = len(rel_doc)
    # If there are no relevant documents, return 0.0
    if rel_num == 0:
        return 0.0
    # iterate over the retrieved doc
    precision_sum = 0.0
    ret_rel_num = 0
    for rank, (document_id, _) in enumerate(results, start=1):
        if document_id in rel_doc:
            ret_rel_num += 1
            precision = ret_rel_num / rank
            precision_sum += precision

    # Initialize variables for precision calculation
    average_precision = precision_sum / rel_num
    return average_precision


# calculate the bpref
def calc_bpref(common_doc, query_result, rel_length, rel_doc, non_rel, ordered_docs):
    bpref = 0
    # iterate over the relevant retrieved documents
    for doc in common_doc:
        # Calculate the bpref for each relevant retrieved doc
        bpref += 1 - first_Rank(query_result, doc, rel_doc, non_rel, rel_length, ordered_docs) / rel_length
    return bpref/rel_length


# Calculate the number of non-relevant docs before a given document
def first_Rank(query_result, doc, rel_doc, un_rel, rel_length, ordered_docs):
    q_r = [item[0] for item in query_result]
    # Find the documents ranked higher than the given one
    higher = set(ordered_docs[:q_r.index(doc)])
    non_relevance = 0
    for d in higher:
        # Count the non-relevant docs number
        if non_relevance == rel_length:
            return rel_length
        if d not in rel_doc and d not in un_rel:
            non_relevance += 1
    return non_relevance


# load and process the results file
def load_results(results_file):
    query_results = {}
    with open(results_file, "r") as file:
        # Read all lines from the results file
        lines = file.readlines()

        for line in lines:
            # Remove whitespace
            line = line.strip()
            if line:
                parts = line.split()
                # query_id
                query_id = parts[0]
                # document_id
                doc_id = parts[2]  # 2
                # similarity_score
                sim = float(parts[3])  # 3

                if query_id not in query_results:
                    query_results[query_id] = []

                # Append the doc id and similarity
                query_results[query_id].append((doc_id, sim))
    return query_results


# Load the qrels file
def load_relevance_judgments(qrels):
    rel = {}
    non_rel = {}

    with open(qrels, "r") as q:
        lines = q.readlines()

        for line in lines:
            # Remove whitespace
            line = line.strip()
            if line:
                query_result = line.split()
                # query_id
                query_id = query_result[0]
                # document_id
                doc_id = query_result[2]
                # relevance
                relevance = int(query_result[3])

                # Judge whether is relevant or non-relevant
                if relevance == 0:
                    if query_id in non_rel:
                        # Add doc id to non-relevant dict
                        non_rel[query_id].add(doc_id)
                    else:
                        # or create a new non-relevant dict
                        non_rel[query_id] = {doc_id}
                else:
                    if query_id in rel:
                        rel[query_id].add(doc_id)
                    else:
                        rel[query_id] = {doc_id}

    return {"relevant": rel, "non_relevant": non_rel}


# Calculate the values
def calc_evaluation_values(results_file, qrels_file):
    relevance_judgments = load_relevance_judgments(qrels_file)
    query_results = load_results(results_file)

    # get relevant and non-relevant docs
    rel = relevance_judgments["relevant"]
    non_relevant = relevance_judgments["non_relevant"]

    num_queries = len(query_results)
    value = {}

    # Initialize metric counters
    total_precision = 0.0
    total_recall = 0.0
    total_p_at_10 = 0.0
    total_r_precision = 0.0
    total_average_precision = 0.0
    total_bpref = 0.0

    for query_id, results in query_results.items():
        # relevant and non-relevant docs
        rel_doc = rel[query_id]
        non_doc = {}

        # Convert results to a dict
        results_dict = {doc_id: score for doc_id, score in results}
        # Find common documents between query results and relevant documents
        common_docs = set(results_dict.keys()) & set(rel_doc)
        # Sort query results
        ordered_docs1 = sorted(results, key=lambda x: x[1], reverse=True)
        ordered_docs = [item[0] for item in ordered_docs1]

        num_rel_doc = len(rel_doc)
        # calc metrics for the current query
        precision = calc_precision(results, rel_doc)
        recall = calc_recall(results, rel_doc)
        p_at_10 = calc_p_at_10(results, rel_doc)
        r_precision = calc_r_precision(results, rel_doc, num_rel_doc)
        average_precision = calc_average_precision(results, rel_doc)
        bpref = calc_bpref(common_docs, results, num_rel_doc, rel_doc, non_doc, ordered_docs)

        # Update metric counters
        total_precision += precision
        total_recall += recall
        total_p_at_10 += p_at_10
        total_r_precision += r_precision
        total_average_precision += average_precision
        total_bpref += bpref

    # calc average metrics
    value['Precision'] = total_precision / num_queries
    value['Recall'] = total_recall / num_queries
    value['P@10'] = total_p_at_10 / num_queries
    value['R-Precision'] = total_r_precision / num_queries
    value['MAP'] = total_average_precision / num_queries
    value['bpref'] = total_bpref / num_queries

    # return metrics
    print("Evaluation results:")
    for name, value in value.items():
        print(f"{name}: {value:}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="BM25 Information Retrieval")
    # parser.add_argument("-m", "--mode", choices=["interactive", "automatic"], required=True,
    #                     help="Specify the mode: interactive or automatic")
    # args = parser.parse_args()

    results_file = "./results.txt"
    qrels_file = "files/qrels.txt"

    # Check whether the result file exists
    if os.path.exists(results_file):
        # Calculate results
        calc_evaluation_values(results_file, qrels_file)
    else:
        print("Results created and saved successfully.")


