from langchain.prompts import PromptTemplate

medical_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly knowledgeable medical assistant.

Use the provided medical context to answer the question accurately.
If the answer is not present in the context, say:
"I am not sure based on the provided medical documents."

Context:
{context}

Question:
{question}

Answer:
"""
)
