!pip install ibm-watsonx-ai==0.2.6
!pip install langchain==0.1.16
!pip install langchain-ibm==0.1.4
!pip install transformers==4.41.2
!pip install huggingface-hub==0.23.4
!pip install sentence-transformers==2.5.1
!pip install chromadb
!pip install wget==3.2
!pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

filename = 'companyPolicies.txt'
ur = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

wget.download(url, out=filename)
print('file downloaded')

with open(filename, 'r') as file:
    contents = file.read()
    print(contents)
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
print('document ingested')

model_id = 'google/flan-t5-xl'
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEX_TOKENS: 130,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5
}

credentials = {
    "url": "url",
    "api_key": "key"
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_llm = WatsonLLM(model=model)

qa = RetrievalQA.from_chain_type(llm=flan_llm,chain_type="stuff",retriever=docsearch.as_retriever(),
                                 return_sourse_documents=False)
query= "what is mobile policy?"
qa.invoke(query)

------------------------------------------------------
model_id = 'meta-llama/llama-3-3-70b-instruct'

parameters = {
    GenParams.DECODINGMETHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5
}

credentials = {
    "url": "url"
}
project_id = "project"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

llama_llm = WatsonxLLM(model=model)

qa = RetrievalQA.from_chain_type(llm=llama_llm, chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=False)
query = "Can you summarize the document for me?"
qa.invoke(query)

prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.
{context}
Question: {Question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=llama_llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs,
                                 return_source_documents=False)
query = "Can i eat in company vehicles?"
qa.invoke(query)

memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
qa = ConversationalRetrievalChain.from_llm(llm=llama_llm, chain_type="stuff",
                                           retriever=docsearch.as_retriever(),
                                           memory=memory,
                                           get_chat_history=lambda h : h,
                                           return_resource_documents=False)
history = []
query = "What is mobile policy?"
result = qa.invoke({"question":query}, {"chat_history":history})
print(result["answer"])
history.append((query, result["answer"]))
result = qa({"question": query}, {"chat_history": history})

# An Agent
def qa():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(llm=llama_llm, chain_type="stuff",
    retriever=docsearch.as_retriever(), memory = memory, get_chat_history=lambda h:h,
    return_source_documents=False)

    history = []
    while True:
        query = input("Question: ")

        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = qa({"question": query}, {"chat_history": history})
        history.append((query, result["answer"]))
        print("Answer: ", result["answer"])
