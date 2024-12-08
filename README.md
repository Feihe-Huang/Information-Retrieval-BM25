# COMP3009J Information Retrieval 

**How to use**\
Remember to go to the corresponding folder and run the code.

Users can select the appropriate mode by using command line arguments to run the search for small corpus:
- python search_small_corpus.py -m interactive


Or for automatic mode:
- python search_small_corpus.py -m automatic


User can run the evaluation program to test the small corpus using the following command:
- python evaluate_small_corpus.py


Users can select the appropriate mode by using command line arguments to run the search for large corpus:
- python search_large_corpus.py -m interactive


Or for automatic mode:
- python search_large_corpus.py -m automatic


User can run the evaluation program to test the large corpus using the following command:
- python evaluate_large_corpus.py

**Output file Format:**\

**Extract documents strategy:**\
First I convert terms to lower case. Then I divide the document into terms using regular expression which only allows the lower case letters and numbers.


**index.txt:**\
This file is used to store the index which is suitable for
performing retrieval later.


The file contains the term, documents which contain the term and the BM25 value.
I store the BM25 value in it so that it won't be calculated in the retrieval process which can improve the retrieve speed and efficiency. 


**results.txt:**\
I write all the results into this file rather than the first 15 results as it is not required.
This file will have 4 fields on each line, which are:
1. The Query ID.
2. The Document ID.
3. The rank of the document in the results for this query (starting at 1).
4. The similarity score for the document and this query.

### Results/Speed
#####search_small_corpus.py:
######When index is not exist(Automatic Mode):
Index cost: 0.419145 s


Query cost: 0.24564600000000003 s

######When index is already exist(Automatic Mode):
Load index cost: 0.062182 s


Query cost: 0.252224 s


#####search_large_corpus.py

######When index is not exist(Automatic Mode):
Index cost: 35.546121 s

Query cost: 0.3865239999999943 s

######When index is already exist(Automatic Mode):
Load index cost: 3.874669 s

Query cost: 0.3617899999999996 s

######Evaluation results for small corpus:
- Precision: 0.010514650836855821
- Recall: 0.9440939159230575
- P@10: 0.28044444444444433
- R-Precision: 0.3541536088066606
- MAP: 0.3850511743637609
- bpref: 0.3354013158486026


######Evaluation results for large corpus:
- Precision: 0.020053899084978937
- Recall: 0.9970909243998191
- P@10: 0.5592592592592591
- R-Precision: 0.5158084892146382
- MAP: 0.5734456059641256
- bpref: 0.5603904648490999

