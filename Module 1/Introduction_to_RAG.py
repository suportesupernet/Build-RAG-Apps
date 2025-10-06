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

