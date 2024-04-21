#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: Apache-2.0
#

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import git
import os
from queue import Queue

# from prompts import model_prompt, custom_question_prompt

from prompts_llama3 import model_prompt, custom_question_prompt

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
allowed_extensions = [".py", ".ipynb", ".md"]

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split("/")[-1]
        self.clone_path = last_name.split(".")[0]
        self.MyQueue = Queue(maxsize=2)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(
                            os.path.join(dirpath, file), encoding="utf-8"
                        )
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass
        self.delete_directory(self.clone_path)
        return self.docs

    def chunk_files(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200
        )
        chunked_documents = text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def get_conversation_chain(self, gen_ai_key):
        credentials = Credentials(api_key=gen_ai_key)
        client = Client(credentials=credentials)
        # Create vector db
        docs = self.extract_all_files()
        chunked_documents = self.chunk_files(docs)
        embeddings = LangChainEmbeddingsInterface(
            client=client,
            model_id="sentence-transformers/all-minilm-l6-v2",
            parameters=TextEmbeddingParameters(truncate_input_tokens=True),
        )
        vector_store = FAISS.from_documents(
            documents=chunked_documents, embedding=embeddings
        )

        retriever = vector_store.as_retriever()
        search_kwargs = {"k": 3}

        retriever.search_kwargs.update(search_kwargs)

        #  Create conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        prompt_template = model_prompt()
        qa_chain_prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )
        question_prompt = PromptTemplate.from_template(custom_question_prompt())
        #  Create conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        credentials = Credentials(
            api_key=gen_ai_key,
            api_endpoint="https://bam-api.res.ibm.com/v2/text/chat?version=2024-03-19",
        )
        client = Client(credentials=credentials)

        llm = LangChainChatInterface(
            model_id="meta-llama/llama-3-70b-instruct",
            client=client,
            parameters=TextGenerationParameters(
                decoding_method=DecodingMethod.GREEDY,
                max_new_tokens=2040,
                min_new_tokens=10,
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                return_options=TextGenerationReturnOptions(
                    input_text=False, input_tokens=True
                ),
            ),
            moderations=ModerationParameters(
                # Threshold is set to very low level to flag everything (testing purposes)
                # or set to True to enable HAP with default settings
                hap=ModerationHAP(input=True, output=False, threshold=0.01)
            ),
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": qa_chain_prompt},
            condense_question_prompt=question_prompt,
            verbose=True,
        )
        self.delete_directory(self.clone_path)
        return conversation_chain

    def retrieve_results(self, query, conversation_chain):
        chat_history = list(self.MyQueue.queue)
        # qa = self.get_conversation_chain(vector_store, gen_ai_key)
        result = conversation_chain({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result["answer"]
