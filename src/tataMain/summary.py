from src.tataMain.utils import img,tab,NarrativeText
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import base64
from langchain_core.messages import HumanMessage
from src.tataMain.logger import logging
import json

logging.info("Summerization Start")
load_dotenv()
AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

img_file_path = "summary/images.json"
tab_file_path = "summary/tables.json"
narrative_file_path = "summary/narrative_text.json"
img_base64_list_file_path = "summary/img_base64_list.json"

if os.path.exists(img_file_path) and os.path.exists(tab_file_path) and os.path.exists(narrative_file_path):
    logging.info("Files Exist")
    logging.info("Loading data from files")
    with open(img_file_path, 'r') as img_file:
        image_sum = json.load(img_file)
    with open(tab_file_path, 'r') as tab_file:
        table_summaries = json.load(tab_file)
    with open(narrative_file_path, 'r') as narrative_file:
        text_summaries = json.load(narrative_file)
    with open(img_base64_list_file_path, 'r') as img_base64_list_file:
        img_base64_list = json.load(img_base64_list_file)

else:        
   prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
      These summaries will be embedded and used to retrieve the raw table elements. \
      Give a concise summary of the table that is well optimized for retrieval. Table:{element} """

   prompt = ChatPromptTemplate.from_template(prompt_text)


   llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0,
   )

   summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

   table_summaries = []

   table_summaries = summarize_chain.batch(tab, {"max_concurrency": 16})

   prompt = ChatPromptTemplate.from_template(prompt_text)

   model = AzureChatOpenAI(
      azure_deployment="gpt-4o",
      temperature=0,
    )
   summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

   text_summaries = []

   text_summaries = summarize_chain.batch(NarrativeText, {"max_concurrency": 16})


   def encode_image(image_path):
      """Getting the base64 string"""
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
   def image_summarize(img_base64, prompt):
     """Make image summary"""
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


   def generate_img_summaries(path):
      """
      Generate summaries and base64 encoded strings for images
      path: Path to list of .jpg files extracted by Unstructured
      """

      # Store base64 encoded images
      img_base64_list = []

      # Store image summaries
      image_summaries = []

      # Prompt
      prompt = """You are an assistant tasked with summarizing images for retrieval. \
      These summaries will be embedded and used to retrieve the raw image. \
      Give a concise summary of the image that is well optimized for retrieval."""

      # Apply to images
      for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))


      return img_base64_list, image_summaries

   fpath=r"D:\tata\extracted_data"

# Image summaries
   img_base64_list, image_summaries = generate_img_summaries(fpath)

   image_sum= image_summaries[:85]

   logging.info("Summerization End")
   logging.info("Saving extracted data to files")
   with open(img_file_path, 'w') as img_file:
        json.dump(image_sum, img_file)
   with open(tab_file_path, 'w') as tab_file:
        json.dump(table_summaries, tab_file)
   with open(narrative_file_path, 'w') as narrative_file:
        json.dump(text_summaries, narrative_file)
   with open(img_base64_list_file_path, 'w') as img_base64_list_file:
        json.dump(img_base64_list, img_base64_list_file)
