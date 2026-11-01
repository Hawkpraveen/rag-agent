from contextlib import contextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main
from rag import Ollama


class UsageTests(unittest.TestCase):
    def test_records_actual_embedding_and_chat_counts(self):
        model = Ollama()
        replies = iter([
            {"embeddings": [[1, 0]], "prompt_eval_count": 5},
            {"message": {"content": "21 days [1]."}, "model": "resolved-model",
             "prompt_eval_count": 60, "eval_count": 8},
        ])
        model.post = lambda *_: next(replies)
        model.embed(["refund"])
        model.answer("refund?", [{"id": 1, "text": "21 days"}])
        self.assertEqual([(u['input_tokens'], u['output_tokens']) for u in model.usage], [(5, 0), (60, 8)])
        self.assertEqual(model.usage[1]['model_name'], 'resolved-model')

    def test_failed_citation_still_records_tokens_and_missing_counts_are_unknown(self):
        model = Ollama()
        model.post = lambda *_: {"message": {"content": "Invalid [9]"}, "eval_count": 3}
        with self.assertRaisesRegex(RuntimeError, 'citation'):
            model.answer("refund?", [{"id": 1, "text": "21 days"}])
        self.assertIsNone(model.usage[0]['input_tokens'])
        self.assertEqual(model.usage[0]['output_tokens'], 3)

    def test_ask_api_returns_usage_on_success_and_failure(self):
        usage = [{"operation": "chat", "input_tokens": 50, "output_tokens": 9}]
        for error in [None, RuntimeError('Invalid citation')]:
            with self.subTest(error=error):
                def answer(*_):
                    if error:
                        raise error
                    return {"answer": "21 days", "sources": []}
                @contextmanager
                def session():
                    yield SimpleNamespace(ask=answer, client=SimpleNamespace(usage=usage))
                with patch.object(main, 'agent_session', session):
                    response = main.ask(main.QuestionRequest(question='refund?'))
                if error:
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(json.loads(response.body)['token_usage'], usage)
                else:
                    self.assertEqual(response['token_usage'], usage)


if __name__ == '__main__':
    unittest.main()
