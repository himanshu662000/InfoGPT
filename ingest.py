import os
import shutil
import logging
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    JSONLoader
)
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    filename='document_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def document_loader_with_splitting(doc_dir, file_type_mappings):
    """
    Load and split documents using the provided parameters.

    Parameters:
    - doc_dir (str): Document directory to be loaded.
    - file_type_mappings (dict): A dictionary mapping file extensions to loader classes.

    Returns:
    dict: A dictionary containing split texts for each document type.
    """
    logging.info(f"Loading and splitting documents in directory: {doc_dir}")
    logging.info(f"File type mappings: {file_type_mappings}")

    texts = []
    chunk_size = int(os.environ.get("CHUNK_SIZE", "1024"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))
    logging.info(f"chunk size:{chunk_size} and chunk overlap:{chunk_overlap}")

    for glob_pattern, loader_cls in file_type_mappings.items():
        try:
            logging.info(f"Processing files with pattern: {glob_pattern}")
            loader_kwargs = { 'jq_schema':'.', 'text_content':False } if loader_cls == JSONLoader else None
            loader_dir = DirectoryLoader(
                doc_dir, glob=glob_pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs)
            documents = loader_dir.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            # for different glob pattern it will split and add texts
            texts += text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(
                f"An error occurred while processing files with pattern {glob_pattern}: {e}")
            # Handle the error or continue to the next file type

    return texts  # Returing texts only because in embeding it takes with the glob_pattern


def doc_load_n_split_doctype():
    """
        Returns:
    list: A list of split texts for each document.
    """
    file_type_mappings = {
        '*.txt': TextLoader,
        '*.pdf': PyPDFLoader,
        '*.csv': CSVLoader,
        '*.docx': Docx2txtLoader,
        '*.xlss': UnstructuredExcelLoader,
        '*.html': UnstructuredHTMLLoader,
        '*.pptx': UnstructuredPowerPointLoader,
        '*.ppt': UnstructuredPowerPointLoader,
        '*.md': UnstructuredMarkdownLoader,
        '*.json': JSONLoader,
    }
    # Below code to define the path of document sources

    doc_dir = os.path.join(os.getcwd(), "document_sources")
    all_doc_dir = [doc_dir]

    for dirpath, dirnames, filenames in os.walk(doc_dir):
        for dirname in dirnames:
            all_doc_dir.append(os.path.join(dirpath, dirname))
    
    texts = []

    for doc_dir in all_doc_dir:
        texts += document_loader_with_splitting(doc_dir, file_type_mappings)

    return texts


def delete_db_folder(db_folder_path):
    try:
        if os.path.exists(db_folder_path):
            shutil.rmtree(db_folder_path)
            logging.info(f"The '{db_folder_path}' folder has been deleted.")
        else:
            logging.info(f"The '{db_folder_path}' folder does not exist.")
    except Exception as e:
        logging.error(f"An error occurred while deleting '{db_folder_path}': {e}")

if __name__ == "__main__":

    load_dotenv()

    # Define the path to the db folder
    db_folder_path = "db"

    delete_db_folder(db_folder_path)

    texts = doc_load_n_split_doctype()

    model_name = os.environ.get("EMBEDING_MODEL","thenlper/gte-base")
    # set True to compute cosine similarity
    encode_kwargs = {'normalize_embeddings': True}

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    # Here is the nmew embeddings being used
    embedding = model_norm

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=db_folder_path)

    logging.info(f"The '{db_folder_path}' folder has been created.")

