import os
import pandas as pd
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === GPU Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Paths Configuration ===
RUN_FILE_NAME = "WholeQ_RM3_RETRIEVAL_T2022.txt"
output_base_dir = "runs/ZERO_SHOT"
RUN_NAME = "Deepseek_Zero_Shot"
os.makedirs(output_base_dir, exist_ok=True)

trec_collection_path = "data/clinicaltrials/2023/corpus.jsonl"
topics_csv = "data/2022/ct_queries.tsv"
retrieved_trials_file = f"runs/FIRST_STAGE/{RUN_FILE_NAME}"
retrieval_txt_path = os.path.join(output_base_dir, f"{RUN_FILE_NAME.split('.')[0]}_{RUN_NAME.lower()}.txt")

# === Global Configurations ===
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 10
USE_BFLOAT16 = False
FLOAT_TYPE = torch.float16

# === Load Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=FLOAT_TYPE,
)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=FLOAT_TYPE,
    device_map="auto"
)

print("Model loaded successfully.")

# === Load Corpus ===
def load_corpus():
    corpus_dict = {}
    with open(trec_collection_path, "r", encoding="utf-8") as f:
        for line in f:
            trial = json.loads(line.strip())
            corpus_dict[trial["id"]] = trial.get("contents")
    print(f"Loaded {len(corpus_dict)} documents.")
    return corpus_dict

corpus_cache = load_corpus()

# === Scoring ===
def score_trial(query, trial_text, topic_no=None, trial_id=None):
    if trial_text == "Text not found." or trial_text is None:
        print(f"Missing trial text for trial ID: {trial_id}")
        return 0.0

    prompt = f""" 
    ### Role: You are an expert in biomedical AI with access to clinical trial data and the ability to assess the relevance of a given patient case description to a specific clinical trial. Your task is to evaluate whether the trial is relevant to the patient case.
    
    ### Instruction: 
    - Given a patient description and a clinical trial, assign a unique relevance score between 0 and 1.
    - Higher scores indicate greater relevance to the query.
    - Do not provide explanations or reasoning.

    ### Patient Description: {query}

    ### Clinical Trial: {trial_text}

    ### Output Format:
    A single floating-point number between 0 and 1 only. No text, no explanations, no additional information.

    ### Output:
    """

    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True
        )

    raw_output = TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
    score_text = raw_output.split('Output:')[-1]
    match = re.findall(r"\b(0\.\d{1,5}|1\.0{1,5})\b", score_text)
    if match:
        score = float(match[-1])
    else:
        print(f"Invalid output for Topic {topic_no}, Trial {trial_id}: '{score_text}'")
        score = 0.0

    print(f"Topic {topic_no} | Trial {trial_id} | Score: {score:.5f}")
    return score

# === Read Queries and First-Stage Retrieval ===
df = pd.read_csv(topics_csv, sep="\t", header=None, names=["id", "text"])

bm25_results = {}
with open(retrieved_trials_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        topic_no, trial_id = parts[0], parts[2]
        if topic_no not in bm25_results:
            bm25_results[topic_no] = []
        bm25_results[topic_no].append(trial_id)

# === Rerank and Save ===
text_lines = []
for topic_no, trials in tqdm(bm25_results.items(), desc="Processing Topics", position=0):
    topic_query = df[df['id'].astype(str) == topic_no]["text"].values[0]
    trial_scores = []

    for trial_id in tqdm(trials, desc=f"Scoring Trials for Topic {topic_no}", position=1, leave=False):
        trial_text = corpus_cache.get(trial_id)
        score = score_trial(topic_query, trial_text, topic_no, trial_id)
        trial_scores.append((trial_id, score))

    trial_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (trial_id, score) in enumerate(trial_scores[:100], start=1):
        text_line = f"{topic_no} Q0 {trial_id} {rank} {score:.5f} {RUN_NAME}"
        text_lines.append(text_line)

with open(retrieval_txt_path, "w") as txt_file:
    txt_file.write("\n".join(text_lines) + "\n")

print(f"Saved results to {retrieval_txt_path}")
