from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import (
    OPENAI_MODEL_NAME,
    OPENAI_MODEL_TEMPERATURE,
)
from vectorstore import get_vectorstore
from prompts import contextualize_prompt, qa_prompt


# =========================
# Memory (equivalente ao helper antigo)
# =========================

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns a chat history associated with a given session id.

    If the session id is not present in the store, creates a new
    InMemoryChatMessageHistory and stores it.

    :param session_id: The session id to retrieve the chat history from.
    :return: The chat history associated with the given session id.
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


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
            "input": RunnablePassthrough(),
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
            "context": lambda x: x["context"],
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
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

    Returns a chain that mimics the behavior of create_retrieval_chain, returning
    a dictionary with "answer", "context", and "input" keys.

    :return: A RAG chain that takes in an input and returns a dictionary with answer,
             context and input.
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

    # Chain que captura input, context e gera answer
    # Retorna um dicionário similar ao create_retrieval_chain
    rag_chain = RunnablePassthrough.assign(
        context=lambda x: history_aware_retriever.invoke(
            {"input": x["input"], "chat_history": x.get("chat_history", [])}
        )
    ).assign(answer=qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# =========================
# Exemplo de uso (obrigatório)
# =========================
if __name__ == "__main__":
    chain = get_rag_chain()

    response = chain.invoke(
        {"input": "Explique o contrato"},
        config={"configurable": {"session_id": "user-123"}},
    )

    print("Resposta completa:", response)
    print("\n" + "=" * 50)
    print("Answer:", response["answer"])
    print("\n" + "=" * 50)
    print("Context (documentos):", response["context"])
