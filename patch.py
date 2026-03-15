with open("src/ai_workers/workers/vl_reranker.py") as f:
    code = f.read()

# Replace images.append(self._load_image(query_image_url))
# with query_image_obj if provided
old_block = """        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            images.append(self._load_image(query_image_url))"""

new_block = """        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            if query_image_obj is not None:
                images.append(query_image_obj)
            else:
                images.append(self._load_image(query_image_url))"""

code = code.replace(old_block, new_block)

# Replace the loop in rerank
old_loop = """            # Score each document against the query
            results = []
            for i, doc in enumerate(body.documents):
                if isinstance(doc, str):
                    doc_text = doc
                    doc_image = None
                else:
                    doc_text = doc.text
                    doc_image = doc.image_url

                score = self._score_pair(
                    body.model,
                    body.query,
                    doc_text,
                    query_image_url=body.query_image_url,
                    document_image_url=doc_image,
                )
                results.append(RerankResult(index=i, relevance_score=score, document=doc_text))"""

new_loop = """            # Pre-download query image if provided to avoid redundant network requests
            query_image_obj = None
            if body.query_image_url:
                query_image_obj = self._load_image(body.query_image_url)

            # Score each document against the query
            results = []
            for i, doc in enumerate(body.documents):
                if isinstance(doc, str):
                    doc_text = doc
                    doc_image = None
                else:
                    doc_text = doc.text
                    doc_image = doc.image_url

                score = self._score_pair(
                    body.model,
                    body.query,
                    doc_text,
                    query_image_url=body.query_image_url,
                    document_image_url=doc_image,
                    query_image_obj=query_image_obj,
                )
                results.append(RerankResult(index=i, relevance_score=score, document=doc_text))"""

code = code.replace(old_loop, new_loop)

with open("src/ai_workers/workers/vl_reranker.py", "w") as f:
    f.write(code)

print("Patch applied")
