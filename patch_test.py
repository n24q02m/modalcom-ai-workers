with open("tests/test_workers_vl_reranker.py") as f:
    code = f.read()

# Replace the test test_rerank_with_query_image_url to mock _load_image
old_test = """def test_rerank_with_query_image_url(server):
    server._score_pair = MagicMock(return_value=0.7)

    with patch.dict(os.environ, {"API_KEY": "k"}):"""

new_test = """def test_rerank_with_query_image_url(server):
    server._score_pair = MagicMock(return_value=0.7)
    server._load_image = MagicMock(return_value="mock_image_obj")

    with patch.dict(os.environ, {"API_KEY": "k"}):"""

code = code.replace(old_test, new_test)

with open("tests/test_workers_vl_reranker.py", "w") as f:
    f.write(code)

print("Patch applied to test")
