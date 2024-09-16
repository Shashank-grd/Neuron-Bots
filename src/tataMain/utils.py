from unstructured.partition.pdf import partition_pdf
from src.tataMain.logger import logging

logging.info("Extracting start")
raw_pdf_element=partition_pdf(
    filename=r"D:\tata\data\dashboard.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,
    extract_image_block_types=["Image","Table"],
    extract_image_block_to_payload=False,
    extract_image_block_output_dir="extracted_data"
)

logging.info("Extracting Completed")
img=[]
for element in raw_pdf_element:
  if "unstructured.documents.elements.Image" in str(type(element)):
    img.append(str(element))

tab=[]
for element in raw_pdf_element:
  if "unstructured.documents.elements.Table" in str(type(element)):
    tab.append(str(element))


NarrativeText =[]
for element in raw_pdf_element:
  if "unstructured.documents.elements.NarrativeText" in str(type(element)):
    NarrativeText.append(str(element))

