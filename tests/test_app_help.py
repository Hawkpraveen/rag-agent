import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch

from main import ask, QuestionRequest

from app_help import app_help_answer
from rag import Agent


class AppHelpTests(unittest.TestCase):
    def test_api_returns_help_without_sources_or_token_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'RAG_DB': str(Path(directory) / 'index.sqlite3')}):
                response = ask(QuestionRequest(question='how to upload a doc ?'))
        self.assertEqual(response['answer_type'], 'app_help')
        self.assertEqual(response['sources'], [])
        self.assertEqual(response['token_usage'], [])

    def test_upload_help_bypasses_retrieval_and_models_even_with_empty_library(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            agent = Agent(Path(directory) / 'index.sqlite3', model)
            try:
                for question in ('how to upload a doc ?', 'How do I attach a PDF?',
                                 'Where can I add documents?', 'Can you show me how to import files?'):
                    result = agent.ask(question)
                    self.assertEqual(result['answer_type'], 'app_help')
                    self.assertEqual(result['sources'], [])
                    self.assertIn('10 MB', result['answer'])
                    self.assertIn('Add documents', result['answer'])
                self.assertEqual(model.mock_calls, [])
            finally:
                agent.db.close()

    def test_document_questions_still_go_to_rag(self):
        for question in ('What does the upload policy say?',
                         'According to the manual, how do I upload a file?',
                         'How do I upload a document according to the handbook?',
                         'Summarize my uploaded documents',
                         'How do I add files in the document?',
                         'Compare upload procedures across my documents'):
            with self.subTest(question=question):
                self.assertIsNone(app_help_answer(question))
