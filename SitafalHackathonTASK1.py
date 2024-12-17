import pdfplumber

def extract_text_from_pdf(pdf_path, pages):
    extracted_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            extracted_text[page_num] = pdf.pages[page_num].extract_text()
    return extracted_text

pdf_path = r"C:/Users/ramav/OneDrive/Desktop/TASK 1/task01.py"
pages_to_extract = [1, 9] 
extracted_text = extract_text_from_pdf(pdf_path, pages_to_extract)

print("Page 2 Content:\n", extracted_text[1])
print("Page 6 Content:\n", extracted_text[9])



from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


chunks_page2 = chunk_text(extracted_text[1])
chunks_page6 = chunk_text(extracted_text[9])


embeddings_page2 = embedding_model.encode(chunks_page2)
embeddings_page6 = embedding_model.encode(chunks_page6)



import faiss
import numpy as np

print("Embeddings for Page 2:", embeddings_page2)
print("Shape of Embeddings for Page 2:", embeddings_page2.shape)
print("Embeddings for Page 6:", embeddings_page6)
print("Shape of Embeddings for Page 6:", embeddings_page6.shape)



dimension = embeddings_page2.shape[1]  
index = faiss.IndexFlatL2(dimension)


index.add(np.array(embeddings_page2))  
index.add(np.array(embeddings_page6)) 

print(f"Total chunks in FAISS index: {index.ntotal}")




def retrieve_relevant_chunks(query, embedding_model, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks


query = "What is the unemployment rate for a Bachelor's degree?"
relevant_chunks = retrieve_relevant_chunks(query, embedding_model, index, chunks_page2)

print("Relevant Chunks:\n", relevant_chunks)



import openai


openai.api_key = "your_openai_api_key"


def generate_response_with_llm(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Based on the following information:\n{context}\n\nAnswer the question: {query}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()


response = generate_response_with_llm(query, relevant_chunks)
print("Generated Response:\n", response)




comparison_query = "Compare unemployment rates for Bachelor's and Associate degrees."
comparison_chunks = retrieve_relevant_chunks(comparison_query, embedding_model, index, chunks_page2)


comparison_response = generate_response_with_llm(comparison_query, comparison_chunks)
print("Comparison Response:\n", comparison_response)