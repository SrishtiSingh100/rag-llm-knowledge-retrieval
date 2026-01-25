import os
from flask import Flask, render_template, request, jsonify
from src.helper import load_vector_store, create_embeddings

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths & embeddings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

embeddings = create_embeddings()
db = load_vector_store(FAISS_PATH, embeddings)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")

    if not user_question:
        return jsonify({"answer": "Please provide a question."})

    # -----------------------------
    # Retrieve top relevant documents
    # -----------------------------
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
    # ✅ NEW (works)
    docs = retriever.invoke(user_question)
    
    context = "\n".join([doc.page_content for doc in docs])
    answer = f"Context from medical documents:\n{context}"

    return jsonify({"answer": answer})

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
