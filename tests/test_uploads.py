import asyncio
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

from fastapi import HTTPException
from starlette.requests import Request
import main
from rag import Agent

class Model:
    embedding_model = 'test-uploads'
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

def request(name, content, declared=None):
    headers = [(b'content-type', b'application/octet-stream'), (b'x-filename', quote(name).encode())]
    if declared is not None:
        headers.append((b'content-length', str(declared).encode()))
    async def receive():
        return {'type': 'http.request', 'body': content, 'more_body': False}
    return Request({'type': 'http', 'method': 'POST', 'path': '/upload', 'headers': headers}, receive)

class UploadTests(unittest.TestCase):
    def test_file_is_embedded_retrievable_and_repeated_upload_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            @contextmanager
            def session():
                agent = Agent(root / 'index.sqlite3', Model())
                try:
                    yield agent
                finally:
                    agent.db.close()
            with patch.object(main, 'ROOT', root), patch.object(main, 'agent_session', session):
                body = b'Refunds are available for 21 days.'
                first = asyncio.run(main.upload(request('policy.txt', body)))
                second = asyncio.run(main.upload(request('policy.txt', body)))
                self.assertEqual(first['indexed_documents'], 1)
                self.assertEqual(second['indexed_documents'], 0)
                with session() as agent:
                    hits = agent.retrieve('refunds')
                self.assertEqual(hits[0]['text'], body.decode())
                self.assertEqual(Path(hits[0]['source']).read_bytes(), body)

    def test_empty_unsupported_oversized_and_path_traversal_are_rejected(self):
        for name, content, declared, expected in [
                ('empty.txt', b'', None, 400), ('bad.exe', b'bad', None, 415),
                ('../secret.txt', b'bad', None, 415), ('..\\secret.txt', b'bad', None, 415),
                ('large.txt', b'body', 10 * 1024 * 1024 + 1, 413)]:
            with self.subTest(name=name), self.assertRaises(HTTPException) as error:
                asyncio.run(main.upload(request(name, content, declared)))
            self.assertEqual(error.exception.status_code, expected)

if __name__ == '__main__':
    unittest.main()
