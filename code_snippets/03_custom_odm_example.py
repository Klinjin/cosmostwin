from llm_engineering.domain.documents import ArticleDocument, UserDocument, PDFDocument

if __name__ == "__main__":
    user = UserDocument.get_or_create(first_name="Linda", last_name="Jin")
    articles = PDFDocument.bulk_find(author_id=str(user.id))

    print(f"User ID: {user.id}")  # noqa
    print(f"User name: {user.first_name} {user.last_name}")  # noqa
    print(f"Number of articles: {len(articles)}")  # noqa
    print("First article link:", articles[0].link)  # noqa
c
