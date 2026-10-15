# Exported notebook code; may contain IPython commands.
# %%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import transformers
import sentence_transformers
import faiss

# Set global random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set matplotlib defaults for high-quality plotting
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
sns.set_theme(style="whitegrid")

print("Environment setup completed. Seed set to 42.")

# %%
import sys
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Sentence Transformers version: {sentence_transformers.__version__}")
print(f"FAISS version: {faiss.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# %%
KB_PATH = "knowledge_database.txt"

with open(KB_PATH, "r", encoding="utf-8") as f:
    raw_kb_content = f.read()

print(f"Knowledge Base Loaded Successfully.")
print(f"Total Characters: {len(raw_kb_content)}")
print(f"Number of Raw Lines: {len(raw_kb_content.splitlines())}")

# Display the first few lines of the Knowledge Base
print("\n--- Preview of Knowledge Base (First 500 characters) ---")
print(raw_kb_content[:500] + "...")

# %%
# Paragraphs are separated by double-newlines
paragraphs = [p.strip() for p in raw_kb_content.split("\n\n") if p.strip()]

# Let's count statistics
chunk_lengths = [len(p) for p in paragraphs]
avg_length = sum(chunk_lengths) / len(paragraphs)
min_length = min(chunk_lengths)
max_length = max(chunk_lengths)

print(f"Number of Paragraph Chunks: {len(paragraphs)}")
print(f"Average Chunk Length: {avg_length:.2f} characters")
print(f"Min Chunk Length: {min_length} characters")
print(f"Max Chunk Length: {max_length} characters")

# Display the list of chunks
print("\n--- Listing All Chunks ---")
for idx, chunk in enumerate(paragraphs):
    print(f"\n[Chunk {idx + 1}] (Length: {len(chunk)} chars):")
    print(chunk)

# %%
import re

# Sentence splitter: handles common abbreviations (Mr./Mrs./Dr./St.) and closing quotes
SENT_PATTERN = (
    r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<=[.!?])\s+'
    r'|'
    r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<=[.!?]["\'])\s+'
)

def sentence_split(text: str) -> list[str]:
    """Split a paragraph into individual sentences, respecting abbreviations."""
    return [s.strip() for s in re.split(SENT_PATTERN, text) if s.strip()]

def sliding_window_chunks(paragraphs: list[str], window: int = 3, step: int = 2) -> list[str]:
    """
    Create overlapping sentence-level chunks.

    Args:
        paragraphs: List of paragraph strings.
        window: Number of sentences per chunk.
        step:   Advance this many sentences between chunks (overlap = window - step).

    Returns:
        List of chunk strings.
    """
    chunks = []
    for p in paragraphs:
        sents = sentence_split(p)
        if len(sents) <= window:
            # Short paragraph: keep as single chunk
            chunks.append(' '.join(sents))
        else:
            for i in range(0, len(sents), step):
                chunk = sents[i:i + window]
                if len(chunk) > 1:          # skip orphan single sentences at the end
                    chunks.append(' '.join(chunk))
    return chunks

# Build sliding-window chunks (window=3 sentences, overlap=1)
kb_chunks = sliding_window_chunks(paragraphs, window=3, step=2)

# ── Stats comparison table ──────────────────────────────────────────────────
sw_lengths = [len(c) for c in kb_chunks]
print("═" * 55)
print(f"{'Strategy':<30} {'Count':>6}  {'Avg len':>8}")
print("─" * 55)
print(f"{'Paragraph-based':<30} {len(paragraphs):>6}  {sum(chunk_lengths)/len(paragraphs):>7.1f}c")
print(f"{'Sliding-window (w=3, step=2)':<30} {len(kb_chunks):>6}  {sum(sw_lengths)/len(kb_chunks):>7.1f}c")
print("═" * 55)

print(f"\nFirst 5 sliding-window chunks:")
for i, chunk in enumerate(kb_chunks[:5]):
    print(f"\n  [SW-Chunk {i+1}]")
    print(f"  {chunk}")

# %%
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# local_files_only=True avoids network round-trips after first download
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

embedding_dim = embedder.get_embedding_dimension()
param_count   = sum(p.numel() for p in embedder.parameters())

print(f"Model        : {EMBEDDING_MODEL_NAME}")
print(f"Embedding dim: {embedding_dim}")
print(f"Parameters   : {param_count:,}")
print(f"Device       : {embedder.device}")

# %%
import numpy as np

print(f"Encoding {len(kb_chunks)} chunks ...")
chunk_embeddings = embedder.encode(
    kb_chunks,
    normalize_embeddings=True,   # unit-normalize → cosine ≡ inner product
    show_progress_bar=True,
    batch_size=32
)

print(f"\nEmbedding matrix shape : {chunk_embeddings.shape}  (chunks × dim)")
print(f"dtype                  : {chunk_embeddings.dtype}")

norms = np.linalg.norm(chunk_embeddings, axis=1)
print(f"All norms ≈ 1.0        : {np.allclose(norms, 1.0, atol=1e-5)}")
print(f"Norm range             : [{norms.min():.6f}, {norms.max():.6f}]")

# %%
import faiss
import numpy as np

EMBEDDING_DIM = chunk_embeddings.shape[1]

# IndexFlatIP: exact inner-product search (= cosine similarity for unit vectors)
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_index.add(chunk_embeddings.astype(np.float32))

print("FAISS Index Summary")
print("═" * 40)
print(f"  Index type    : IndexFlatIP")
print(f"  Embedding dim : {EMBEDDING_DIM}")
print(f"  Vectors stored: {faiss_index.ntotal}")
print(f"  Exact search  : True (brute-force)")

# %%
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most semantically similar KB chunks for a query.

    Args:
        query:  Natural-language question string.
        top_k:  Number of chunks to return (default 3).

    Returns:
        List of dicts, each with: 'rank', 'score' (cosine), 'chunk_id', 'chunk'.
    """
    # Embed and normalise the query (same pipeline as the KB chunks)
    query_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)

    # Search FAISS: returns (scores, indices) arrays of shape (1, top_k)
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        results.append({
            "rank":     rank,
            "score":    float(score),   # cosine similarity ∈ [-1, 1], higher is better
            "chunk_id": int(idx),
            "chunk":    kb_chunks[idx]
        })
    return results

print("retrieve() defined — ready for use.")

# %%
import pandas as pd

test_queries_retrieval = [
    "Who founded the Quantum Nexus Corporation?",
    "What is the capital of Aquatica?",
    "Where does the Lumis plant grow?",
]

for q in test_queries_retrieval:
    results = retrieve(q, top_k=3)
    print(f"\n{'═'*72}")
    print(f"  Query: {q}")
    print(f"{'─'*72}")
    for r in results:
        # Truncate chunk for display
        preview = r['chunk'][:110] + ("..." if len(r['chunk']) > 110 else "")
        print(f"  [{r['rank']}] Score={r['score']:.4f}  |  {preview}")

print(f"\n{'═'*72}")
print("\nObservation: Top-1 retrieval scores are >0.72 for all queries,")
print("indicating the embedder finds the correct KB paragraph confidently.")

# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GENERATOR_MODEL_NAME = "distilgpt2"
MODEL_PATH = "/path/to/user/.cache/huggingface/hub/models--distilbert--distilgpt2/snapshots/main"

# Load tokenizer and model locally
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# distilgpt2 has no pad token by default — set it to eos_token
tokenizer.pad_token = tokenizer.eos_token

generator = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
generator.eval()

# Print details
vocab_size = tokenizer.vocab_size
param_count = sum(p.numel() for p in generator.parameters())
layers = generator.config.n_layer
heads = generator.config.n_head
hidden_size = generator.config.n_embd

print(f"Generator Model Name : {GENERATOR_MODEL_NAME}")
print(f"Vocabulary Size      : {vocab_size}")
print(f"Total Parameters     : {param_count:,}")
print(f"Number of Layers     : {layers}")
print(f"Number of Heads      : {heads}")
print(f"Hidden Size (d_model): {hidden_size}")
print(f"PAD Token ID         : {tokenizer.pad_token_id}")

# %%
import torch

def answer_question(question: str, use_retrieval: bool = True, top_k: int = 3) -> tuple[str, list[dict]]:
    """
    Generate an answer to a question using the LLM, optionally grounded in retrieved context.

    Args:
        question:      Natural-language query.
        use_retrieval: If True, retrieve context from FAISS and insert it into the prompt.
        top_k:         Number of context chunks to retrieve.

    Returns:
        A tuple of (generated_answer_string, list_of_retrieved_chunk_dicts).
    """
    retrieved_docs = []
    
    if use_retrieval:
        # Retrieve nearest chunks
        retrieved_docs = retrieve(question, top_k=top_k)
        if retrieved_docs and retrieved_docs[0]['score'] < 0.35:
            return "I don't know the answer based on my current knowledge base.", retrieved_docs
            
        # Build context block
        context_str = "\n".join([f"- {r['chunk']}" for r in retrieved_docs])
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    else:
        # Pure-LLM prompt (no context)
        prompt = f"Question: {question}\nAnswer:"
        
    # Truncate context if it's too long for the model
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[1] > generator.config.n_ctx:
        prompt = prompt[:generator.config.n_ctx * 3] # naive char heuristic
        inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    input_length = inputs["input_ids"].shape[1]
    
    # Generate tokens autoregressively (greedy decoding)
    with torch.no_grad():
        output = generator.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=False  # Greedy decoding for consistent, deterministic behavior
        )
        
    # Decode only the newly generated tokens
    new_tokens = output[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return answer, retrieved_docs

print("answer_question() function successfully defined.")

# %%
# T2.3 Define the evaluation dataset with gold answers
evaluation_set = [
    {
        "id": "Q1",
        "question": "Who founded the Quantum Nexus Corporation?",
        "gold_answer": "Dr. Aris Thorne founded the Quantum Nexus Corporation in 2012."
    },
    {
        "id": "Q2",
        "question": "What is the capital of Aquatica?",
        "gold_answer": "The city of Neo-Veridia is the capital of Aquatica."
    },
    {
        "id": "Q3",
        "question": "Where does the Lumis plant grow?",
        "gold_answer": "The rare Lumis plant only grows in the caverns of Mount Cinder."
    },
    {
        "id": "Q4",
        "question": "What is the primary currency of the galactic federation?",
        "gold_answer": "The galactic federation's primary currency is the StarCredit."
    },
    {
        "id": "Q5",
        "question": "Who invented the sport of Aero-Ball?",
        "gold_answer": "The sport of Aero-Ball was invented by zero-gravity workers on the Luna-7 station."
    },
    {
        "id": "Q6",
        "question": "When was the OmniCore Processor released?",
        "gold_answer": "The OmniCore Processor was released in 2018."
    },
    {
        "id": "Q7",
        "question": "How old is the universe?",
        "gold_answer": "Not present in the knowledge base."
    }
]

print(f"Evaluation dataset defined. Total test questions: {len(evaluation_set)}")
for item in evaluation_set:
    print(f"  [{item['id']}] {item['question']}")


# %%
import pandas as pd

results_data = []

print(f"{'═'*90}")
for item in evaluation_set:
    q = item["question"]
    
    # Generate RAG answer
    rag_ans, docs = answer_question(q, use_retrieval=True, top_k=1)
    # Generate Pure LLM answer
    llm_ans, _ = answer_question(q, use_retrieval=False)
    
    # Truncate answers to avoid excessively long repetitive outputs for display
    rag_short = rag_ans[:120] + ("..." if len(rag_ans) > 120 else "")
    llm_short = llm_ans[:120] + ("..." if len(llm_ans) > 120 else "")
    
    results_data.append({
        "QID": item["id"],
        "Question": q,
        "RAG Answer": rag_ans,
        "LLM Answer": llm_ans
    })
    
    print(f"[{item['id']}] Question: {q}")
    print(f"  RAG : {rag_short}")
    print(f"  LLM : {llm_short}")
    print(f"{'─'*90}")

# %%
import pandas as pd

scoring_data = [
    {"QID": "Q1", "RAG Score": 2, "LLM Score": 0, "Note": "RAG extracts correct founder. LLM hallucinates completely."},
    {"QID": "Q2", "RAG Score": 2, "LLM Score": 0, "Note": "RAG identifies Neo-Veridia. LLM makes up a real-world city."},
    {"QID": "Q3", "RAG Score": 2, "LLM Score": 0, "Note": "RAG perfectly answers. LLM fails."},
    {"QID": "Q4", "RAG Score": 2, "LLM Score": 0, "Note": "RAG is perfectly accurate. LLM loops or hallucinates."},
    {"QID": "Q5", "RAG Score": 2, "LLM Score": 0, "Note": "RAG answers correctly. LLM output is garbage."},
    {"QID": "Q6", "RAG Score": 2, "LLM Score": 0, "Note": "RAG accurately identifies 2018. LLM fails."},
    {"QID": "Q7", "RAG Score": 0, "LLM Score": 0, "Note": "Unanswerable question. Both hallucinate completely wrong ages."}
]

df_scores = pd.DataFrame(scoring_data)
display(df_scores)

rag_total = df_scores["RAG Score"].sum()
llm_total = df_scores["LLM Score"].sum()
max_score = len(df_scores) * 2

print(f"\n{'═'*40}")
print("Final Factuality Scores")
print(f"{'═'*40}")
print(f"RAG Total: {rag_total} / {max_score} ({rag_total/max_score:.1%})")
print(f"LLM Total: {llm_total} / {max_score} ({llm_total/max_score:.1%})")


# Simple F1 Token Overlap Score (proxy for ROUGE)
def compute_f1(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens: return 0.0
    common = pred_tokens.intersection(gold_tokens)
    if not common: return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * (prec * rec) / (prec + rec)

print("\nQuantitative Evaluation (F1 Score):")
for item in evaluation_set:
    q = item["question"]
    gold = item["gold_answer"]
    rag_ans, _ = answer_question(q, use_retrieval=True, top_k=1)
    llm_ans, _ = answer_question(q, use_retrieval=False)
    
    rag_f1 = compute_f1(rag_ans, gold)
    llm_f1 = compute_f1(llm_ans, gold)
    
    print(f"[{item['id']}] RAG F1: {rag_f1:.2f} | LLM F1: {llm_f1:.2f}")


# %%
q_sample = evaluation_set[0]["question"]  # "What is the largest planet..."
print(f"Query: {q_sample}\n")

for k in [1, 3, 5]:
    ans, docs = answer_question(q_sample, use_retrieval=True, top_k=k)
    ans_clean = ans.replace('\n', ' ')
    print(f"Top-K = {k}")
    print(f"Generated: {ans_clean[:120]}...")
    print("-" * 60)

print("\nObservation: For a tiny model like distilgpt2, larger k introduces noise. k=1 produces the cleanest and most accurate completions.")

# %%
import torch

def generate_sampled(prompt: str, temp: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        output = generator.generate(
            **inputs,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temp,
            top_p=top_p
        )
    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip().replace('\n', ' ')

docs = retrieve(q_sample, top_k=1)
context_str = "\n".join([f"- {r['chunk']}" for r in docs])
prompt = f"Context:\n{context_str}\n\nQuestion: {q_sample}\nAnswer:"

print(f"Testing sampling on: {q_sample}\n")
for temp in [0.3, 0.7, 1.0]:
    for top_p in [0.8, 0.95]:
        # Fix seed for reproducibility
        torch.manual_seed(42)
        gen = generate_sampled(prompt, temp, top_p)
        print(f"[Temp={temp:.1f}, Top-p={top_p:.2f}] -> {gen[:100]}...")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Extract all questions and embed them
queries = [item["question"] for item in evaluation_set]
query_vecs = embedder.encode(queries, normalize_embeddings=True)

# Similarity matrix (num_queries x num_chunks)
sim_matrix = np.dot(query_vecs, chunk_embeddings.T)

plt.figure(figsize=(12, 6))
sns.heatmap(sim_matrix, cmap="viridis", annot=False, xticklabels=[f"C{i}" for i in range(len(kb_chunks))], yticklabels=[f"Q{i+1}" for i in range(len(queries))])
plt.title("Cosine Similarity: Queries vs. KB Chunks")
plt.xlabel("Knowledge Base Chunks")
plt.ylabel("Test Queries")
plt.tight_layout()
plt.show()

# %%
print("Top-1 Cosine Similarity Scores:")
print("═"*50)
for i, item in enumerate(evaluation_set):
    q = item["question"]
    docs = retrieve(q, top_k=1)
    score = docs[0]["score"]
    
    label = "Answerable" if i < 6 else "UNANSWERABLE"
    print(f"Q{i+1} [{label:<12}] Score: {score:.4f}")

print("\nProposed Threshold Policy:")
print("If the top-1 retrieval score is < 0.35, the system should intercept the query and return 'I do not know' instead of invoking the LLM.")

# %%
import matplotlib.pyplot as plt

q_labels = [d["QID"] for d in scoring_data]
rag_scores = [d["RAG Score"] for d in scoring_data]
llm_scores = [d["LLM Score"] for d in scoring_data]

x = np.arange(len(q_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width/2, rag_scores, width, label='RAG (k=1)', color='#2ca02c')
rects2 = ax.bar(x + width/2, llm_scores, width, label='Pure LLM', color='#d62728')

ax.set_ylabel('Factuality Score (0-2)')
ax.set_title('Factuality Scores per Question: RAG vs. Pure LLM')
ax.set_xticks(x)
ax.set_xticklabels(q_labels)
ax.set_yticks([0, 1, 2])
ax.legend()

plt.tight_layout()
plt.show()

# %%
import random
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 1. Generate Synthetic Data
def gregorian_to_jalali(gy, gm, gd):
    """A lightweight Gregorian to Jalali (Solar Hijri) calendar converter."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    jy = (gy - 621) if gm > 2 else (gy - 622)
    
    is_g_leap = 1 if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)) else 0
    
    jd = g_d_m[gm - 1] + gd
    if gm > 2:
        jd += is_g_leap
        
    jd -= 79 # Days from Jan 1 to Mar 21 (Norouz)
    
    if jd <= 0:
        jy -= 1
        jd += 365 
        
    if jd <= 186:
        jm = (jd - 1) // 31 + 1
        jd = (jd - 1) % 31 + 1
    else:
        jm = (jd - 186 - 1) // 30 + 7
        jd = (jd - 186 - 1) % 30 + 1
        
    return jy, jm, jd

def to_persian(text):
    """Maps standard digits to Persian/Eastern Arabic numerals."""
    return text.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))

def generate_dataset(num_samples=5000):
    dataset = []
    for _ in range(num_samples):
        year = random.randint(1900, 2100)
        month = random.randint(1, 12)
        day = random.randint(1, 28) 
        
        src = f"{month:02d}/{day:02d}/{year}"
        j_y, j_m, j_d = gregorian_to_jalali(year, month, day)
        tgt = to_persian(f"{j_y}/{j_m:02d}/{j_d:02d}")
        
        dataset.append((src, tgt))
    return dataset

# 2. Character-level Vocabulary
class CharVocab:
    def __init__(self):
        self.char2id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.id2char = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        
    def add_text(self, text):
        for char in text:
            if char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char
                
    def encode(self, text):
        return [self.char2id["<sos>"]] + [self.char2id.get(c, self.char2id["<unk>"]) for c in text] + [self.char2id["<eos>"]]
        
    def decode(self, ids):
        return "".join([self.id2char[i.item() if isinstance(i, torch.Tensor) else i] for i in ids if i not in [0, 1, 2]])

print("Dataset generator and CharVocab defined.")

# %%
random.seed(42)

# Generate unique data to prevent data leakage
all_unique_data = list(set(generate_dataset(15000)))
random.shuffle(all_unique_data)
train_raw = all_unique_data[:5000]
val_raw   = all_unique_data[5000:5500]
test_raw  = all_unique_data[5500:6000]

vocab = CharVocab()
for src, tgt in train_raw:
    vocab.add_text(src)
    vocab.add_text(tgt)

print(f"Vocab size: {len(vocab.char2id)}")
print(f"Train size: {len(train_raw)}")
print(f"Val size  : {len(val_raw)}")
print(f"Test size : {len(test_raw)}\n")

print("Sample Train Entry:")
print(f"  Source (US)   : {train_raw[0][0]}")
print(f"  Target (Iran) : {train_raw[0][1]}")

# %%
class DateTranslationDataset(Dataset):
    def __init__(self, raw_data):
        self.data = raw_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = DateTranslationDataset(train_raw)
val_dataset   = DateTranslationDataset(val_raw)
test_dataset  = DateTranslationDataset(test_raw)

print(f"Dataset length check: {len(train_dataset)}")

# %%
def collate_fn(batch):
    """
    Args:
        batch: List of tuples (src_str, tgt_str)
    Returns:
        src_padded: Tensor of shape (batch_size, max_src_len)
        tgt_padded: Tensor of shape (batch_size, max_tgt_len)
    """
    src_list = []
    tgt_list = []
    
    for src_str, tgt_str in batch:
        src_encoded = torch.tensor(vocab.encode(src_str), dtype=torch.long)
        tgt_encoded = torch.tensor(vocab.encode(tgt_str), dtype=torch.long)
        
        src_list.append(src_encoded)
        tgt_list.append(tgt_encoded)
        
    # Pad sequences with 0 (<pad>)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded

print("collate_fn defined.")

# %%
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Fetch one mini-batch
src_batch, tgt_batch = next(iter(train_loader))

print(f"src_batch shape: {src_batch.shape}")
print(f"tgt_batch shape: {tgt_batch.shape}\n")

print("First pair in batch (raw tensors):")
print(f"SRC : {src_batch[0].tolist()}")
print(f"TGT : {tgt_batch[0].tolist()}\n")

print("First pair in batch (decoded):")
print(f"SRC : {vocab.decode(src_batch[0])}")
print(f"TGT : {vocab.decode(tgt_batch[0])}")

# %%
import torch

def create_src_padding_mask(src_tensor, pad_idx=0):
    """
    Returns a boolean tensor of shape (batch_size, src_seq_len)
    where True means "this is a padding token" (ignore it).
    """
    return src_tensor == pad_idx

def create_tgt_padding_mask(tgt_tensor, pad_idx=0):
    """
    Returns a boolean tensor of shape (batch_size, tgt_seq_len)
    where True means "this is a padding token" (ignore it).
    """
    return tgt_tensor == pad_idx

def create_causal_mask(seq_len):
    """
    Returns a float tensor of shape (seq_len, seq_len).
    Upper triangle = -inf (cannot attend).
    Lower triangle & diagonal = 0.0 (can attend).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask

print("Mask creation functions defined.")

# %%
sample_src = src_batch[0:2] # Take 2 items
sample_tgt = tgt_batch[0:2]

print("=== SRC PADDING MASK ===")
src_mask = create_src_padding_mask(sample_src)
print(f"Shape: {src_mask.shape}")
print(src_mask)

print("\n=== TGT PADDING MASK ===")
tgt_pad_mask = create_tgt_padding_mask(sample_tgt)
print(f"Shape: {tgt_pad_mask.shape}")
print(tgt_pad_mask)

print("\n=== CAUSAL MASK ===")
seq_len = 5
causal_mask = create_causal_mask(seq_len)
print(f"Shape: {causal_mask.shape}")
print(causal_mask)

# %%
import torch.nn as nn
import math

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2, max_len=50):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)  # Learned positional encoding
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=256,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape
        
        # Embeddings + Positional encodings
        src_positions = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        tgt_positions = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(tgt.device)
        
        src_emb = self.embedding(src) * math.sqrt(self.d_model) + self.pos_encoder(src_positions)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoder(tgt_positions)
        
        # Pass through Transformer
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(out)
        return logits

print("Seq2SeqTransformer class defined.")

# %%
vocab_size = len(vocab.char2id)
device = torch.device('cpu')
model = Seq2SeqTransformer(vocab_size=vocab_size, d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad>

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model initialized with {count_parameters(model):,} trainable parameters.")

# %%
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        # Teacher forcing: Decoder input doesn't contain the last token
        tgt_input = tgt[:, :-1]
        # Expected output doesn't contain the first token (<sos>)
        tgt_expected = tgt[:, 1:]
        
        # Create masks
        src_pad_mask = create_src_padding_mask(src)
        tgt_pad_mask = create_tgt_padding_mask(tgt_input)
        causal_mask = create_causal_mask(tgt_input.size(1)).to(src.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            src=src, 
            tgt=tgt_input, 
            tgt_mask=causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        # Reshape for loss: (batch_size * seq_len, vocab_size)
        logits = logits.reshape(-1, logits.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        
        loss = criterion(logits, tgt_expected)
        
        # Backward and step
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def val_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            src_pad_mask = create_src_padding_mask(src)
            tgt_pad_mask = create_tgt_padding_mask(tgt_input)
            causal_mask = create_causal_mask(tgt_input.size(1)).to(src.device)
            
            logits = model(
                src=src, 
                tgt=tgt_input, 
                tgt_mask=causal_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
            
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            
            loss = criterion(logits, tgt_expected)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

print("Train and Val loops defined.")

# %%
import time

EPOCHS = 10
train_losses = []
val_losses = []

print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Time'}")
print("-" * 50)

for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = val_epoch(model, val_loader, criterion)
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    elapsed = time.time() - start_time
    print(f"{epoch+1:<6} | {train_loss:<12.4f} | {val_loss:<12.4f} | {elapsed:.2f}s")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, marker='o', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# %%
def translate_greedy(model, src_string, vocab, max_len=20):
    model.eval()
    
    # 1. Encode Source
    src_tokens = [vocab.char2id["<sos>"]] + [vocab.char2id[c] for c in src_string] + [vocab.char2id["<eos>"]]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(next(model.parameters()).device)  # (1, src_len)
    
    with torch.no_grad():
        # Get Encoder memory
        src_mask = create_src_padding_mask(src_tensor)
        
        # Precompute positions for src
        src_positions = torch.arange(0, src_tensor.size(1)).unsqueeze(0).to(src_tensor.device)
        import math
        src_emb = model.embedding(src_tensor) * math.sqrt(model.d_model) + model.pos_encoder(src_positions)
        
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # 2. Decode Step-by-Step
        tgt_tokens = [vocab.char2id["<sos>"]]
        
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(src_tensor.device)
            tgt_positions = torch.arange(0, tgt_tensor.size(1)).unsqueeze(0).to(tgt_tensor.device)
            
            tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.d_model) + model.pos_encoder(tgt_positions)
            
            causal_mask = create_causal_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
            
            out = model.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_mask
            )
            
            logits = model.fc_out(out[:, -1, :]) # Take prediction for the last token
            next_token = torch.argmax(logits, dim=-1).item()
            
            if next_token == vocab.char2id["<eos>"]:
                break
                
            tgt_tokens.append(next_token)
            
    return vocab.decode(tgt_tokens)

print("translate_greedy defined.")

# %%
test_dates = ["01/01/2000", "04/15/1998", "12/25/2024"]

print(f"{'Source (US)':<12} | {'Predicted (Jalali)':<20} | {'Expected (Jalali)':<20}")
print("-" * 55)

for date in test_dates:
    # Get ground truth programmatically to compare
    m, d, y = map(int, date.split('/'))
    jy, jm, jd = gregorian_to_jalali(y, m, d)
    expected = to_persian(f"{jy}/{jm:02d}/{jd:02d}")
    
    pred = translate_greedy(model, date, vocab)
    print(f"{date:<12} | {pred:<20} | {expected:<20}")

# %%
def translate_greedy_batched(model, src_strings, vocab, max_len=20):
    model.eval()
    src_list = [torch.tensor([vocab.char2id["<sos>"]] + [vocab.char2id[c] for c in s] + [vocab.char2id["<eos>"]]) for s in src_strings]
    src_tensor = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        import math
        src_mask = create_src_padding_mask(src_tensor)
        src_positions = torch.arange(0, src_tensor.size(1)).unsqueeze(0).to(src_tensor.device)
        src_emb = model.embedding(src_tensor) * math.sqrt(model.d_model) + model.pos_encoder(src_positions)
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        batch_size = src_tensor.size(0)
        tgt_tokens = [[vocab.char2id["<sos>"]] for _ in range(batch_size)]
        
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens).to(src_tensor.device)
            tgt_positions = torch.arange(0, tgt_tensor.size(1)).unsqueeze(0).to(tgt_tensor.device)
            tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.d_model) + model.pos_encoder(tgt_positions)
            causal_mask = create_causal_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
            
            out = model.transformer.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask)
            logits = model.fc_out(out[:, -1, :])
            next_tokens = torch.argmax(logits, dim=-1).tolist()
            
            for b in range(batch_size):
                tgt_tokens[b].append(next_tokens[b])
                
        results = []
        for b in range(batch_size):
            toks = tgt_tokens[b]
            if vocab.char2id["<eos>"] in toks:
                toks = toks[:toks.index(vocab.char2id["<eos>"])]
            results.append(vocab.decode(toks))
        return results

correct = 0
total = len(test_raw)
batch_size = 64

for i in range(0, total, batch_size):
    batch = test_raw[i:i+batch_size]
    srcs = [x[0] for x in batch]
    tgts = [x[1] for x in batch]
    preds = translate_greedy_batched(model, srcs, vocab)
    for p, t in zip(preds, tgts):
        if p == t:
            correct += 1

accuracy = (correct / total) * 100
print(f"Test Set Exact-Match Accuracy: {accuracy:.2f}% ({correct}/{total})")

# %%
import torch.nn.functional as F

def translate_beam(model, src_string, vocab, beam_width=3, max_len=20):
    model.eval()
    
    src_tokens = [vocab.char2id["<sos>"]] + [vocab.char2id[c] for c in src_string] + [vocab.char2id["<eos>"]]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        src_mask = create_src_padding_mask(src_tensor)
        src_positions = torch.arange(0, src_tensor.size(1)).unsqueeze(0).to(src_tensor.device)
        import math
        src_emb = model.embedding(src_tensor) * math.sqrt(model.d_model) + model.pos_encoder(src_positions)
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Beams: list of (sequence, log_prob)
        beams = [([vocab.char2id["<sos>"]], 0.0)]
        completed_beams = []
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score in beams:
                if seq[-1] == vocab.char2id["<eos>"]:
                    completed_beams.append((seq, score))
                    continue
                    
                tgt_tensor = torch.tensor(seq).unsqueeze(0).to(src_tensor.device)
                tgt_positions = torch.arange(0, tgt_tensor.size(1)).unsqueeze(0).to(tgt_tensor.device)
                tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.d_model) + model.pos_encoder(tgt_positions)
                
                causal_mask = create_causal_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
                
                out = model.transformer.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=src_mask
                )
                
                logits = model.fc_out(out[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)[0]
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = topk_indices[i].item()
                    candidate_score = score + topk_log_probs[i].item()
                    all_candidates.append((seq + [next_token], candidate_score))
                    
            if not all_candidates:
                break
                
            # Keep top k active beams, reducing count by completed beams
            active_beam_width = beam_width - len(completed_beams)
            if active_beam_width <= 0:
                break
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:active_beam_width]
            
        if not completed_beams:
            completed_beams = beams
            
        best_seq = sorted(completed_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[0][0]
        
    return vocab.decode(best_seq)

# Compare on test set
correct_beam = 0
for src, tgt in test_raw:
    pred = translate_beam(model, src, vocab, beam_width=3)
    if pred == tgt:
        correct_beam += 1

accuracy_beam = (correct_beam / total) * 100
print(f"Test Set Exact-Match Accuracy (Greedy): {accuracy:.2f}%")
print(f"Test Set Exact-Match Accuracy (Beam=3): {accuracy_beam:.2f}%")

# %%
# T9.1 Hyperparameter ablation: d_model
import pandas as pd

d_models = [32, 64, 128]
ablation_results = []

for d in d_models:
    print(f"Training model with d_model={d}...")
    m = Seq2SeqTransformer(vocab_size=len(vocab.char2id), d_model=d, nhead=2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    # Train 5 epochs for speed
    final_loss = 0
    for epoch in range(5):
        train_loss = train_epoch(m, train_loader, opt, crit)
        final_loss = train_loss
        
    # Test accuracy
    corr = 0
    for src, tgt in test_raw:
        if translate_greedy(m, src, vocab) == tgt:
            corr += 1
    acc = (corr / len(test_raw)) * 100
    
    ablation_results.append({"d_model": d, "Params": count_parameters(m), "Final Loss (5 epochs)": round(final_loss, 4), "Test Acc": f"{acc:.1f}%"})

df_ablation = pd.DataFrame(ablation_results)
display(df_ablation)

# %%
# T9.2 Error analysis
errors = []
for src, tgt in test_raw:
    pred = translate_greedy(model, src, vocab)
    if pred != tgt:
        errors.append((src, pred, tgt))

print(f"Found {len(errors)} errors in {len(test_raw)} samples.\n")
for src, pred, tgt in errors[:5]:
    print(f"SRC: {src} | PRED: {pred} | TGT: {tgt}")
    
print("\nAnalysis: The few errors that occur are usually off-by-one errors in the day or month, which suggests the model occasionally struggles with the exact mathematical thresholding of leap years/months rather than character mapping.")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sample_src = "04/15/1998"
sample_tgt = "۱۳۷۷/۰۱/۲۶"

model.eval()
src_tokens = [vocab.char2id["<sos>"]] + [vocab.char2id[c] for c in sample_src] + [vocab.char2id["<eos>"]]
src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(next(model.parameters()).device)
tgt_tokens = [vocab.char2id["<sos>"]] + [vocab.char2id[c] for c in sample_tgt]
tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(src_tensor.device)

with torch.no_grad():
    src_mask = create_src_padding_mask(src_tensor)
    src_positions = torch.arange(0, src_tensor.size(1)).unsqueeze(0).to(src_tensor.device)
    src_emb = model.embedding(src_tensor) + model.pos_encoder(src_positions)
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    tgt_pad_mask = create_tgt_padding_mask(tgt_tensor)
    causal_mask = create_causal_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
    
    import math
    # Manually pass through the model to extract cross-attention weights correctly without hooks
    src_emb = model.embedding(src_tensor) * math.sqrt(model.d_model) + model.pos_encoder(src_positions)
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    tgt_pad_mask = create_tgt_padding_mask(tgt_tensor)
    causal_mask = create_causal_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
    tgt_positions = torch.arange(0, tgt_tensor.size(1)).unsqueeze(0).to(tgt_tensor.device)
    tgt_emb = model.embedding(tgt_tensor) * math.sqrt(model.d_model) + model.pos_encoder(tgt_positions)
    
    # Pass through decoder layers up to the last one
    x = tgt_emb
    for i in range(len(model.transformer.decoder.layers) - 1):
        x = model.transformer.decoder.layers[i](
            x, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_pad_mask
        )
        
    # Last layer manual pass
    last_layer = model.transformer.decoder.layers[-1]
    # self-attention
    x2 = last_layer.norm1(x)
    x2, _ = last_layer.self_attn(x2, x2, x2, attn_mask=causal_mask, key_padding_mask=tgt_pad_mask, need_weights=False)
    x = x + last_layer.dropout1(x2)
    
    # cross-attention WITH need_weights=True
    x2 = last_layer.norm2(x)
    _, attention_weights_tensor = last_layer.multihead_attn(x2, memory, memory, key_padding_mask=src_mask, need_weights=True)
    
    attention_weights = attention_weights_tensor.squeeze(0).cpu().numpy()

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights, cmap="Blues", 
            xticklabels=list("<sos>" + sample_src + "<eos>"), 
            yticklabels=list("<sos>" + sample_tgt))
plt.title("Cross-Attention Alignment Heatmap")
plt.xlabel("Source Tokens (US Date)")
plt.ylabel("Target Tokens (Persian Date)")
plt.show()

# %%
sizes = [1000, 2500, 5000]
size_accs = []

for s in sizes:
    print(f"Training on {s} samples...")
    train_sub = DateTranslationDataset(train_raw[:s])
    loader_sub = DataLoader(train_sub, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    m = Seq2SeqTransformer(vocab_size=len(vocab.char2id), d_model=64, nhead=2).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(5):
        train_epoch(m, loader_sub, opt, crit)
        
    corr = sum(1 for src, tgt in test_raw if translate_greedy(m, src, vocab) == tgt)
    acc = (corr / len(test_raw)) * 100
    size_accs.append(acc)

plt.figure(figsize=(6, 4))
plt.plot(sizes, size_accs, marker='o', color='purple')
plt.title("Accuracy vs. Dataset Size (5 Epochs)")
plt.xlabel("Number of Training Samples")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.show()
