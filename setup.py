from setuptools import find_packages,setup

setup(
    name="Car_Assistant",
    version="0.0.1",
    author="Neuron Bots",
    author_email="shashank.2grd@gmail.com",
    install_reqires=[
        "pillow",
        "pydantic",
        "lxml",
        "matplotlib",
        "unstructured-pytesseract",
        "tesseract-ocr",
        "langchain_core",
        "langchain_openai",
        "langchain",
        "chromadb",
        "langchain_community",
        "fastapi",
        "python-dotenv",
        ],
    packages=find_packages()
)