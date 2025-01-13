import os

from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from llm_engineering.domain.documents import PDFDocument

from .base import BaseCrawler


class CustomPdfCrawler(BaseCrawler):
    model = PDFDocument

    def __init__(self) -> None:
        super().__init__()

    def extract(self, file_path: str, **kwargs) -> None:
        old_model = self.model.find(file_path=file_path)
        if old_model is not None:
            logger.info(f"PDF already exists in the database")

            return

        logger.info(f"Starting scrapping PDF: {file_path}")

        loader = PyPDFLoader(file_path=file_path, extract_images=False)

        docs = []
        docs_lazy = loader.lazy_load()
        for doc in docs_lazy:
            docs.append(doc)

        file_name = os.path.basename(file_path)
        title, _ = os.path.splitext(file_name)

        combined_document = {
            "content": "\n".join([doc.page_content for doc in docs]),  # Combine the text
            "metadata": {**docs[0].metadata, "title": title}
            if docs
            else {},  # Use metadata from the first page (optional)
        }

        content = {
            "Title": combined_document["metadata"].get("title"),
            "Content": combined_document["content"],
        }

        user = kwargs["user"]
        instance = self.model(
            content=content,
            file_path=file_path,
            platform="Phys236",
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping custom PDF: {file_path}")
