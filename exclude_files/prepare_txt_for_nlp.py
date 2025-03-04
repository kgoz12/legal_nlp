# these might be useful when I set up the docker container
# pip install --upgrade pylance --user
# pip install --no-cache-dir llama-cpp-python==0.2.85 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 
from datasets import Dataset, load_dataset
import pandas as pd
from transformers import AutoTokenizer
import pyarrow.dataset
import lance
from lance.vector import vec_to_table
from llama_cpp import Llama
import pyarrow as pa


#########################################################################################################################################################
### LOAD TEXTS THAT ARE AUTHORITATIVE INFO SOURCES ######################################################################################################
#########################################################################################################################################################
# the reference data sources are:
#   1. federalist papers - DONE
#   2. executive orders from prior administrations - DONE
#   3. look for case law / supreme court decisions that are relevant (Marbury v. Madison)?
#   4. constitution?
#   5. Ethics in Government Act of 1978?
#########################################################################################################################################################
#########################################################################################################################################################

# path to the federalist paper files
federalist_txts = load_dataset("text", 
                               data_files={"train": [
                                   "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/federalist-paper-fed47.txt", 
                                   "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/federalist-paper-fed70.txt",
                                   "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/executive_order_12866.txt"
                                   ]})

print(federalist_txts.num_rows) # 3

# let's make this more dynamic later
file_name_list = [
    "federalist-paper-fed47", 
    "federalist-paper-fed70", 
    "executive_order_12866"
    ]

#########################################################################################################################################################
### SPLIT THE TEXTS INTO SHORTER, TOPICALLY HOMOGENEOUS, SEGMENTS #######################################################################################
#########################################################################################################################################################
# the "dumb" way to do this is just tokenize the full text and stop after you've reached N tokens, then pick it up w/ the next N tokens, and so on.
# the "smarter" way to do this is to split the text according to topic. I've struggled to find models that predict the cutpoints in text
# BUT, this model appears to split text by topic and paraphrase so that the syntactical integrity remains intact. Love this:
# https://huggingface.co/unikei/t5-base-split-and-rephrase
# I think I can use this just one for the splitting task.
#########################################################################################################################################################
#########################################################################################################################################################

# #### OPTION A #####
# # send the bodies of the federalist papers through the HF tokenizer (that corresponds to the Llama cpp model I'll use)
# # Model reference: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
# # mixtral-8x7b-instruct-v0.1.Q2_K.gguf

# MAX_LENGTH = 512
# STRIDE = 20

# # load the Mixtral tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "mistralai/Mixtral-8x7B-Instruct-v0.1",
#     max_length = MAX_LENGTH,
#     truncation = True,
#     padding=False,
#     return_overflowing_tokens = True,
#     stride = STRIDE,
#     is_split_into_words = False,
#     return_offsets_mapping = True
# )
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # make the tokenization function
# def tokenize_function(example):
#     return tokenizer(example["text"], 
#                      padding=True, 
#                      padding_side="right",
#                      truncation=True,
#                      max_length = MAX_LENGTH,
#                      return_overflowing_tokens = True,
#                      stride = STRIDE,
#                      return_offsets_mapping = True,
#                      return_tensors="pt")

# tokenized_datasets = federalist_txts["train"].map(tokenize_function, batched=False, remove_columns=None)
# tokenized_datasets = tokenized_datasets.add_column("authoritative_resource", file_name_list)

# # ## Upon inspection, it seems that the stride wasn't implemented
# # ## So, I think I'll need to do something custom to make that happen.

# # drop the vectors we don't need
# tokenized_datasets_subset = tokenized_datasets.remove_columns(['text', 'offset_mapping', 'overflow_to_sample_mapping'])

# # convert the tokenized data to a list so it can later 
# tokenized_datasets_subset_list = tokenized_datasets_subset.to_list()

# # loop through all the vectors in the list and explode them into one long list (w/o nesting vectors)
# exploded_text_list = []
# # loop through the list and generate a dict that 
# for i in range(0, len(tokenized_datasets_subset_list)):
#     for j in range(0, len(tokenized_datasets_subset_list[i]["input_ids"])):
#         text = tokenizer.decode(tokenized_datasets_subset_list[i]["input_ids"][j], skip_special_tokens=True)
#         resource = tokenized_datasets_subset_list[i]["authoritative_resource"]
#         exploded_text_list.append({"authoritative_resource": resource, "text" : text})

# # convert the list to pandas so I can use it later 
# pd_df = pd.DataFrame(exploded_text_list)

#### OPTION B #####
# https://huggingface.co/learn/cookbook/en/advanced_rag
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer
import matplotlib.pyplot as plt 

federalist_txts = federalist_txts["train"].add_column("authoritative_resource", file_name_list)

RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=doc["text"], metadata={"source": doc["authoritative_resource"]}) for doc in federalist_txts]

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# I think we should consider swithcing the tokenzer model
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME ),
    chunk_size=256,
    chunk_overlap=10,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

# N = 245 docs_processed
docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])

# Let's visualize the chunk sizes we would have in tokens from a common model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs_processed]
# fig = pd.Series(lengths).hist()
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.show()

# convert the list to pandas so I can use it later 
exploded_text_list = []
# loop through the list and generate a dict that 
for i in range(0, len(docs_processed)):
    text = docs_processed[i].page_content
    resource = docs_processed[i].metadata["source"]
    exploded_text_list.append({"authoritative_resource": resource, "text" : text})

pd_df = pd.DataFrame(exploded_text_list)

#########################################################################################################################################################
### SEND TEXTS TO SENTENCE EMBEDDINGS MODEL #############################################################################################################
#########################################################################################################################################################
# the "dumb" way to do this is just tokenize the full text and stop after you've reached N tokens, then pick it up w/ the next N tokens, and so on.
# the "smarter" way to do this is to split the text according to topic. I've struggled to find models that predict the cutpoints in text
# BUT, this model appears to split text by topic and paraphrase so that the syntactical integrity remains intact. Love this:
# https://huggingface.co/unikei/t5-base-split-and-rephrase
# I think I can use this just one for the splitting task.
#########################################################################################################################################################
#########################################################################################################################################################
from llama_cpp import Llama

MAX_LENGTH = 512

# https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/blob/main/all-MiniLM-L6-v2.F16.gguf
llm_sentence_embedding = Llama(
    model_path = "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/all-MiniLM-L6-v2.F16.gguf", 
    embedding=True, 
    n_ctx=MAX_LENGTH * 2, 
    verbose=False)

# convert the pd dataframe with the authoritative texts to a list
reference_text_list = pd_df.values.tolist()

# send those texts to the sentence embedding vector model one by one
vector_list = []
for row in reference_text_list:
    vector_list.append(llm_sentence_embedding.embed(row)[0])

# convert the list of vectors to a table then combine that table with the original text and write it to a lance dataset on disk
table = vec_to_table(vector_list)
combined=pa.Table.from_pandas(pd_df).append_column("vector", table["vector"])
ds = lance.write_dataset(combined, "chatbot.lance", mode='overwrite')

# # the lance db needs to have 256 rows to create the vector index
# ds = ds.create_index(
#     "vector", 
#     index_type="IVF_PQ",
#     num_partitions=4,
#     num_sub_vectors=16,
#     replace=True,
#     metric = "L2" # cosine
#     )

# this converts the question text to a vector, this vector then gets compared to the vectors in the database for similarity
# question_text = "What is unitary executive?"
# question_text = "What is the expansion of presidential review over agencies?"
# question_text = "What is a Regulatory Policy Officer?"
question_text = "Does the President provide authoritative interpretations of law for the executive branch?"
question_vector = llm_sentence_embedding.embed([question_text])
print(question_vector[0])

# pull the K most similar authoritative texts to the above question
K = 2
response_data_frame = ds.to_table(
    nearest={"column": "vector",
             "q": question_vector[0],
             "metric": "dot",
             "k": K}).to_pandas()

# vector distances seem biased towards short passages. I need to fix this.
response_data_frame.head(K)

# load the question answering model
llm_q_and_a = Llama(
    "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/mixtral-8x7b-instruct-v0.1.Q2_K.gguf", 
    n_ctx=MAX_LENGTH * 2, 
    n_threads=8, 
    n_gpu_layers=30, 
    verbose=False)

# instead of using rag - ask the question using the context of the top K authoritative texts...
# ... we could ask the question of every text and note the variability in the answers
ado_llm_resp_id = []
ado_llm_resp_impact_analysis_node_text = []
ado_llm_resp_response = []
i = 0
# for row, id in zip(pd_df.text.values, pd_df.id.values):
for row in pd_df.text.values:
    print("***************" + str(i) + "*********************")
    response = llm_q_and_a.create_chat_completion(
        seed=123,
        top_k = 0,
        temperature=0.05,
        max_tokens=512,
        messages = [
            {"role" : "user",
            "content": row},
            {"role": "assistant", 
            "content": "Does the President provide authoritative interpretations of law for the executive branch?"}
        ]
    )
    ado_llm_resp_id.append(str(i))
    ado_llm_resp_impact_analysis_node_text.append(row)
    ado_llm_resp_response.append(response['choices'][0]['message']['content'])
    i = i + 1

for j in range(0, len(ado_llm_resp_response)):
    print("********" + str(j) + "********")
    print(ado_llm_resp_response[j])