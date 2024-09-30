from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.tataMain.multimodal import multi_modal_rag_chain, retriever_multi_vector_img
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import base64
from src.tataMain.logger import logging


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

logging.info("Calling Multimodal rag")
# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)


def i_encode_image(image_bytes: bytes) :
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")

def i_image_summarize(img_base64: str, prompt: str) -> str:
    logging.info("Input Image Summary")
    """Make image summary using Azure OpenAI"""
    chat = AzureChatOpenAI(model="gpt-4o", max_tokens=2000)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_input_img_summary(image_bytes: bytes) :
    """
    Generate a summary and base64 encoded string for a single image
    image_bytes: Bytes of the image file
    """
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Encode the image to base64
    cbase64_image = i_encode_image(image_bytes)

    # Generate a summary for the image
    cimage_summary = i_image_summarize(cbase64_image, prompt)

    return cbase64_image, cimage_summary

@app.post("/api/process_query/")
async def process_query(query: str = Form(...), file: UploadFile = File(None)):
    logging.info("Api hitting")
    """
    Process the user query and image (if provided) and return a response.
    """
    if file:
        # Read image file bytes
        image_bytes = await file.read()
        input_img_base64, input_image_summary = generate_input_img_summary(image_bytes)
        response = chain_multimodal_rag.invoke(
            input_image_summary + " this is the image description. Give answers based on this image: " + query
        )
    else:
        # If no file, just process the query
        response = chain_multimodal_rag.invoke(query)
    logging.info("Send Api Response")
    return JSONResponse(content={"response": response})

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimodal Retrieval API"}