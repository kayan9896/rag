
import os
from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv('.env')
k= os.getenv('key')

# Initialize global variables
conversation_retrieval_chain = None
llm = None
llm_embeddings = None

def init_llm():
    global llm, llm_embeddings
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=k)
    llm_embeddings = OpenAIEmbeddings(openai_api_key = k)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  


# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():

    if 'file' not in request.files:
        return jsonify({
            "botResponse": "The file was not uploaded correctly, Try again or use a different file"
        }), 400

    file = request.files['file']  
    file_path = file.filename  
    file.save(file_path)  

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    init_llm()
    global llm, llm_embeddings, conversation_retrieval_chain
    db = Chroma.from_documents(texts, llm_embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    return jsonify({
        "botResponse": "Thank you for providing your PDF document. You can ask me questions regarding it."
    }), 200

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    prompt = request.json['userMessage']  
    global conversation_retrieval_chain
    result = conversation_retrieval_chain({"question": prompt,"chat_history": []})
    return jsonify({
        "botResponse": result['answer']
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')

