"""app class, the main entrypoint to use FastAPI."""

from fastapi import FastAPI, Request

from messages_buffer import buffer_message

app = FastAPI()


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
        await buffer_message(
            chat_id=chat_id,
            message=message,
        )

    return {"status": "ok"}
