import time
import modal
import sys

from ai_workers.workers.reranker import MODEL_CONFIGS, RerankerServer

server = RerankerServer()
server.load_models = lambda: None
server.models = {}
server.tokenizers = {}

# Test batch scoring logic if we implemented it
