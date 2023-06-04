import gradio as gr
import os
import time
from typing import List, Tuple

from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

def loading_pdf() -> str:
    """
    Function to display loading message when PDF is loading.
    :return: A string indicating that PDF is loading.
    """
    return "Loading..."

def pdf_changes(pdf_files: List[str], open_ai_key: str) -> str:
    """
    Function to process the loaded PDF files.
    :param pdf_files: List of paths to the PDF files.
    :param open_ai_key: OpenAI API key.
    :return: Status message after processing the PDFs.
    """
    if openai_key is not None:
        os.environ['OPENAI_API_KEY'] = open_ai_key

        for pdf_doc in pdf_files:
            loader = OnlinePDFLoader(pdf_doc.name)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1500,
                                                  chunk_overlap=500)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            global qa
            qa = ConversationalRetrievalChain.from_llm(
                llm=OpenAI(temperature=0.5),
                retriever=retriever,
                return_source_documents=False)
        return "Ready"
    else:
        return "You forgot OpenAI API key"

def add_text(history: List[Tuple[str, str]],
             text: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Function to add user input to chat history.
    :param history: Current chat history.
    :param text: User input.
    :return: Updated chat history and an empty string.
    """
    history = history + [(text, None)]
    return history, ""

def bot(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Function to generate bot response and update chat history.
    :param history: Current chat history.
    :return: Updated chat history.
    """
    response = infer(history[-1][0], history)
    history[-1][1] = ""

    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

def infer(question: str, history: List[Tuple[str, str]]) -> str:
    """
    Function to infer the answer from the AI model.
    :param question: User question.
    :param history: Current chat history.
    :return: Answer inferred from the AI model.
    """
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai if ai is not None else "")
        res.append(pair)

    chat_history = res
    query = question
    result = qa({"question": query, "chat_history": chat_history})
    return result["answer"]


css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with documents</h1>
</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        with gr.Column():
            openai_key = gr.Textbox(label="You OpenAI API key", type="password")
            pdf_doc = gr.Files(label="Load a pdf",
                             file_types=['.pdf'],
                             file_count="multiple",
                             type="file")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status",
                                              placeholder="",
                                              interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")

        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question",
                              placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")

    load_pdf.click(loading_pdf, None, langchain_status, queue=False)
    load_pdf.click(pdf_changes,
                   inputs=[pdf_doc, openai_key],
                   outputs=[langchain_status],
                   queue=False)

    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot)

demo.queue(concurrency_count=5, max_size=20).launch()
