"""chain build"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import (
    OPENAI_MODEL_NAME,
    OPENAI_MODEL_TEMPERATURE,
)
from memory import get_session_history
from prompts import contextualize_prompt, qa_prompt
from vectorstore import get_vectorstore

# =========================
# History-aware retriever
# (substitui create_history_aware_retriever)
# =========================


def build_history_aware_retriever(llm, retriever):
    """
    Builds a history-aware retriever by chaining a given language model (LLM) and
    a retriever.

    The history-aware retriever takes in a question and chat history, and rewrites
    the question based on the chat history using the given LLM. The rewritten
    question is then passed to the retriever to generate relevant text snippets.

    :param llm: The language model to use for rewriting the question.
    :param retriever: The retriever to use for generating relevant text snippets.
    :return: A history-aware retriever that takes in a question and chat history, and
             returns relevant text snippets.
    """
    rewrite_question_chain = contextualize_prompt | llm | StrOutputParser()

    return (
        {
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
        }
        | rewrite_question_chain
        | retriever
    )


# =========================
# QA chain (stuff)
# (substitui create_stuff_documents_chain)
# =========================


def build_qa_chain(llm):
    """
    Builds a QA chain by chaining a given language model (LLM) and a prompt.

    The QA chain takes in a context and question, and uses the given LLM to generate an
    answer based on the context and question. The generated answer is then parsed as a
    string using the StrOutputParser.

    :param llm: The language model to use for generating the answer.
    :return: A QA chain that takes in a context and question, and returns an answer.
    """
    return (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )


# =========================
# RAG final
# (substitui create_retrieval_chain)
# =========================


def get_rag_chain():
    """
    Builds a RAG chain by chaining a history-aware retriever and a QA chain.

    The history-aware retriever takes in a context and question, and uses the given LLM to generate
    an answer based on the context and question. The generated answer is then passed to the QA
    chain, which parses the answer as a string.

    The RAG chain takes in a context and question, and returns the parsed answer.

    :return: A RAG chain that takes in a context and question, and returns an answer.
    """
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
    )

    retriever = get_vectorstore().as_retriever()

    history_aware_retriever = build_history_aware_retriever(
        llm,
        retriever,
    )

    qa_chain = build_qa_chain(llm)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: history_aware_retriever.invoke(
                {"question": x["question"], "chat_history": x.get("chat_history", [])}
            )
        )
        | qa_chain
    )

    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# =========================
# Exemplo de uso (obrigatório)
# =========================
if __name__ == "__main__":
    chain = get_rag_chain()

    response = chain.invoke(
        {"question": "Explique o contrato"},
        config={"configurable": {"session_id": "user-123"}},
    )

    print(response)
