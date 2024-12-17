from transformers import pipeline

# Load a pre-trained language model
llm = pipeline('text-generation', model='gpt2')

def generate_response(relevant_chunks, user_query):
    context = "\n".join(relevant_chunks)
    prompt = f"Using the following context: {context}, answer the question: {user_query}"
    response = llm(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].strip()