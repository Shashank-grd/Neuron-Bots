import base64
import uuid
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import io
import re
from src.tataMain.summary import  *
from PIL import Image
import os
from dotenv import load_dotenv
from src.tataMain.logger import logging



logging.info("Multimodal Start")

load_dotenv()
logging.info("Create Multi Vector Retrevier")


AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )


    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
      doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

      summary_docs = [
              Document(page_content=s, metadata={id_key: doc_ids[i]})
              for i, s in enumerate(doc_summaries)
          ]

      retriever.vectorstore.add_documents(summary_docs)
      retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

      # Add texts, tables, and images
      # Check that text_summaries is not empty before adding
    if text_summaries:
          add_documents(retriever,text_summaries,texts)
      # Check that table_summaries is not empty before adding
    if table_summaries:
          add_documents(retriever, table_summaries, tables)
      # Check that image_summaries is not empty before adding
    if image_summaries:
          add_documents(retriever, image_summaries, images)

    return retriever

vectorstore = Chroma(
    collection_name="mm_rag", embedding_function=AzureOpenAIEmbeddings(model="embeddtest")
)

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    NarrativeText,
    table_summaries,
    tab,
    image_sum,
    img_base64_list,
)

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None



def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

logging.info("Split resize or image date")
def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []

    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(600, 300))
            b64_images.append(doc)
        else:
            texts.append(doc)

    return {"images": b64_images, "texts": texts}

logging.info("Image prompt function")
def img_prompt_func(inputs):
    # Limit the number of texts and reduce their content
    max_texts = 2
    max_text_length = 500  
    formatted_texts = "\n".join(
        text[:max_text_length] + "..." if len(text) > max_text_length else text
        for text in inputs["context"]["texts"][-max_texts:]
    )
    
    messages = []
    
    # Limit the number of images
    max_images = 1
    if "images" in inputs["context"] and inputs["context"]["images"]:
        recent_images = inputs["context"]["images"][-max_images:]
        for image in recent_images:
            image_message = {
                "role": "system",
                "content": f"data:image/jpeg;base64,{image}",
            }
            messages.append(image_message)
    
    # Shorten the system message
    text_message = {
        "role": "system",
        "content": (
            "You are a helpful Car assistant.\n"
            "You will be given a mix of information (text and/or images).\n"
            "Use this information to provide relevant details to the user's question.\n"
            "Use information that is given and give answer within the domain of vechile\n"
            "give simple answer and to the point \n"
            f"User-provided question: {inputs['question']}\n\n"
            "Text and/or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    print(messages)
    return messages


logging.info("Multi Modal Rag chain")

def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain with memory
    """

    # Multi-modal LLM
    model = AzureChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=3000)
    
    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain

logging.info("Multimodal End")




