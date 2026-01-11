"""app class, the main entrypoint to use FastAPI."""

from fastapi import FastAPI, Request

from chains import get_rag_chain
from evolution_api import send_whatsapp_message

app = FastAPI()

rag_chain = get_rag_chain()


@app.post("/webhook")
async def webhook(request: Request):
    """
    Handle incoming webhook from external services.

    Returns a JSON response with a single key "status" set to "ok".
    """
    data = await request.json()
    chat_id = data.get("data").get("key").get("remoteJid")
    message = data.get("data").get("message").get("conversation")

    if chat_id and message and "@g.us" not in chat_id:
        ai_response = rag_chain.invoke(
            input={"question": message},
            config={"configurable": {"session_id": chat_id}},
        )["answer"]
        send_whatsapp_message(
            number=chat_id,
            text=ai_response,
        )

    return {"status": "ok"}
