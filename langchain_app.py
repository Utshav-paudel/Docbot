#@ Building document reader fully local
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#@ Loading hugging face embeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings()


def file_to_vdb(FILE_DIRECTORY,embeddings):
    #@ loading files to vector stores
    loader = DirectoryLoader(FILE_DIRECTORY)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)
    splits = text_splitter.split_documents(docs)
    vector_stores = Chroma.from_documents(splits , embeddings)
    return vector_stores

def generate_resonse(llm,vdb,query):
    #@ creating template and loading llm
    template = """Question: {query}\n

    
    Answer : """
    prompt = PromptTemplate(template=template, input_variables=["query"])
    retriever = vdb.as_retriever()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    #@ LLM
    llm = llm
    #@ retriever chain
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return response


def main_runner(vdb,query):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
    model_path="C:/Users/ASUS/.cache/lm-studio/models/TheBloke/phi-2-GGUF/phi-2.Q4_K_S.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
    output = generate_resonse(llm,vdb,query)
    return output
