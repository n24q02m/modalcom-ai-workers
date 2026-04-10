import re

with open("tests/test_workers_mm_reranker.py", "r") as f:
    content = f.read()

# Mocking _load_image, _load_audio, _load_video_frames because they are called before _score_pair now
content = content.replace("def test_rerank_with_query_image(server):", "@patch.object(MmRerankerServer, '_load_image', return_value='img')\ndef test_rerank_with_query_image(mock_load_image, server):")
content = content.replace("def test_rerank_with_doc_images(server):", "@patch.object(MmRerankerServer, '_load_image', return_value='img')\ndef test_rerank_with_doc_images(mock_load_image, server):")
content = content.replace("def test_rerank_with_query_audio(server):", "@patch.object(MmRerankerServer, '_load_audio', return_value=('aud', 16000))\ndef test_rerank_with_query_audio(mock_load_audio, server):")
content = content.replace("def test_rerank_with_doc_audios(server):", "@patch.object(MmRerankerServer, '_load_audio', return_value=('aud', 16000))\ndef test_rerank_with_doc_audios(mock_load_audio, server):")
content = content.replace("def test_rerank_with_query_video(server):", "@patch.object(MmRerankerServer, '_load_video_frames', return_value=['vframe'])\ndef test_rerank_with_query_video(mock_load_video, server):")
content = content.replace("def test_rerank_with_doc_videos(server):", "@patch.object(MmRerankerServer, '_load_video_frames', return_value=['vframe'])\ndef test_rerank_with_doc_videos(mock_load_video, server):")

content = content.replace("def test_rerank_mixed_modalities(server):", "@patch.object(MmRerankerServer, '_load_image', return_value='img')\n@patch.object(MmRerankerServer, '_load_audio', return_value=('aud', 16000))\n@patch.object(MmRerankerServer, '_load_video_frames', return_value=['vframe'])\ndef test_rerank_mixed_modalities(mock_load_video, mock_load_audio, mock_load_image, server):")

with open("tests/test_workers_mm_reranker.py", "w") as f:
    f.write(content)
