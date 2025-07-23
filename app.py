from flask import Flask, request, jsonify
import pickle
import os
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# 📦 Paths
DATA_DIR = "amibot_data"
MODEL_DIR = "./local_model"  # Local MiniLM path

# 🔧 Load preprocessed data
with open(os.path.join(DATA_DIR, "field_variants.pkl"), "rb") as f:
    field_variants = pickle.load(f)

with open(os.path.join(DATA_DIR, "field_map.pkl"), "rb") as f:
    field_map = pickle.load(f)

# 🧠 Load transformer model (local)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_DIR, device=device)

# 🧠 Encode field variants (once)
variant_embeddings = model.encode(
    field_variants, 
    convert_to_tensor=True, 
    batch_size=16, 
    show_progress_bar=False
)

# 🔧 Text cleaner
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# 🤖 Transformer + Fuzzy matching
def get_response(user_input, field_variants, field_map, fuzz_threshold=60):
    cleaned_input = clean_text(user_input)
    query_embedding = model.encode(cleaned_input, convert_to_tensor=True)

    # 🔍 Semantic similarity using cosine
    cosine_scores = util.pytorch_cos_sim(query_embedding, variant_embeddings)[0]
    top_index = torch.argmax(cosine_scores).item()
    top_score = cosine_scores[top_index].item()
    top_variant = field_variants[top_index]

    # 🧪 Fuzzy backup
    fuzzy_score = fuzz.token_set_ratio(cleaned_input, top_variant.lower())

    if fuzzy_score >= fuzz_threshold:
        return {
            "matched": top_variant,
            "semantic_score": round(top_score, 3),
            "fuzzy_score": fuzzy_score,
            "response": field_map[top_variant]
        }
    else:
        return {
            "matched": top_variant,
            "semantic_score": round(top_score, 3),
            "fuzzy_score": fuzzy_score,
            "response": f"🤖 Sorry, I’m not sure what you meant.\n💡 Did you mean: '{top_variant}'?\nPlease rephrase your question."
        }

# 🚀 Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🧠 AmiBot is running with MiniLM!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data.get("query", "")

    if not user_input.strip():
        return jsonify({"error": "Empty query provided."}), 400

    result = get_response(user_input, field_variants, field_map)
    return jsonify(result)

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(debug=True)
