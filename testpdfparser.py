import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = "/Users/link/Desktop/cosmology/PHYS 236 cosmology/lecture/lecture2_FRW.pdf"
loader = PyPDFLoader(file_path=file_path, extract_images=False)

docs = []

docs_lazy = loader.lazy_load()
for doc in docs_lazy:
    docs.append(doc)

# Extract the title from the file name
file_name = os.path.basename(file_path)
title, _ = os.path.splitext(file_name)

"""
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=100000, chunk_overlap=0)

docs = loader.load_and_split(text_splitter=text_splitter)
"""

print(len(docs))  # page number
print(docs[0].page_content[:100])
print(
    docs[0].metadata
)  # {'source': '/Users/link/Desktop/cosmology/PHYS 236 cosmology/lecture/lecture1_overview.pdf', 'page': 5}

combined_document = {
    "content": "\n".join([doc.page_content for doc in docs]),  # Combine the text
    "metadata": {**docs[0].metadata, "title": title} if docs else {},  # Use metadata from the first page (optional)
}

# print(combined_document["content"]) #equations and text
print(
    combined_document["metadata"]
)  # {'source': '/Users/link/Desktop/cosmology/PHYS 236 cosmology/lecture/lecture1_overview.pdf', 'page': 0, 'title': 'lecture2_FRW'}
print(combined_document["metadata"].get("title"))
