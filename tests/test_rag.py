import json
import tempfile
import unittest
from pathlib import Path

from rag import Agent, Ollama, chunks, is_overview_question


class FakeModel:
    embedding_model = "test-model"
    calls = 0
    fail = False

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return [[1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0]
                for text in texts]

    def answer(self, question, sources):
        return "Refunds take 5 days [1]."


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = FakeModel()
        self.agent = Agent(self.root / "index.sqlite3", self.model)
        self.doc = self.root / "refund.md"
        self.doc.write_text("Refunds take 5 days.", encoding="utf-8")

    def tearDown(self):
        self.agent.db.close()
        self.temp.cleanup()

    def test_ingestion_retrieval_and_persistence(self):
        self.agent.ingest(self.doc)
        other = self.root / "support.txt"
        other.write_text("Support is available Monday.", encoding="utf-8")
        self.agent.ingest(other)
        self.agent.db.close()
        self.agent = Agent(self.root / "index.sqlite3", self.model)
        result = self.agent.ask("refund policy")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source"], str(self.doc.resolve()))
        self.assertIn("[1]", result["answer"])

    def test_unchanged_skipped_and_changed_replaced(self):
        self.assertEqual(self.agent.ingest(self.doc), 1)
        self.assertEqual(self.agent.ingest(self.doc), 0)
        self.assertEqual(self.model.calls, 1)
        self.doc.write_text("Refunds take 10 days.", encoding="utf-8")
        self.agent.ingest(self.doc)
        rows = self.agent.db.execute("SELECT content FROM chunks").fetchall()
        self.assertEqual(rows, [("Refunds take 10 days.",)])

    def test_failed_embedding_preserves_existing_document(self):
        self.agent.ingest(self.doc)
        self.doc.write_text("Refunds take 10 days.", encoding="utf-8")
        self.model.fail = True
        with self.assertRaises(RuntimeError):
            self.agent.ingest(self.doc)
        self.assertEqual(self.agent.db.execute("SELECT content FROM chunks").fetchone()[0],
                         "Refunds take 5 days.")

    def test_no_match_and_model_change(self):
        self.agent.ingest(self.doc)
        self.assertEqual(self.agent.ask("support hours")["sources"], [])
        self.model.embedding_model = "changed-model"
        with self.assertRaisesRegex(ValueError, "Embedding model changed"):
            self.agent.retrieve("refund")

    def test_empty_index_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "Index is empty"):
            self.agent.ask("refund")
        with self.assertRaises(ValueError):
            self.agent.retrieve(" ")
        with self.assertRaises(ValueError):
            self.agent.retrieve("refund", top_k=0)

    def test_chunk_coverage_and_overlap(self):
        text = "abcdefghij" * 400
        result = list(chunks(text))
        rebuilt = result[0][1] + "".join(content[200:] for _, content in result[1:])
        self.assertEqual(rebuilt, text)
        self.assertTrue(all(len(content) <= 1400 for _, content in result))

    def test_invalid_model_citation_is_rejected(self):
        model = Ollama()
        model.post = lambda *args: {"message": {"content": "Answer [9]"}}
        with self.assertRaisesRegex(RuntimeError, "citation"):
            model.answer("refund", [{"id": 1, "text": "refund policy"}])

    def test_overview_uses_document_content_without_a_similarity_match(self):
        self.agent.ingest(self.doc)
        other = self.root / "support.txt"
        other.write_text("Support is available Monday.", encoding="utf-8")
        self.agent.ingest(other)
        # A generic question embeds to the support vector in this fake model;
        # nevertheless an overview must cover both documents, including refunds.
        result = self.agent.ask("what does the doc contains ?")
        self.assertEqual({Path(s['source']).name for s in result['sources']},
                         {'refund.md', 'support.txt'})
        self.assertTrue(all(s['score'] is None for s in result['sources']))

    def test_overview_intents_and_specific_question(self):
        for question in ['what does the doc contains ?', 'Summarize my documents',
                         'What is in this file?', 'What is this document about?']:
            self.assertTrue(is_overview_question(question), question)
        self.assertFalse(is_overview_question('What is the refund policy?'))

    def test_overview_prompt_names_sources_and_asks_for_a_summary(self):
        model = Ollama()
        captured = {}
        def post(endpoint, payload):
            captured.update(payload)
            return {'message': {'content': json.dumps({'summaries': [
                {'document_id': 1, 'summary': 'Covers refunds.', 'citations': [1]}
            ]})}}
        model.post = post
        model.answer('what does the doc contains ?',
                     [{'id': 1, 'source': 'support.txt', 'text': 'Refunds take 5 days.'}])
        prompt = captured['messages'][1]['content']
        self.assertIn('Document 1: support.txt', prompt)
        self.assertIn('Excerpt [1]', prompt)
        self.assertIn('Refunds take 5 days.', prompt)

    def test_overview_groups_interleaved_excerpts_and_cites_each_document(self):
        model = Ollama()
        captured = {}
        def post(endpoint, payload):
            captured.update(payload)
            return {'message': {'content': json.dumps({'summaries': [
                {'document_id': 1, 'summary': 'Covers support and loyalty.', 'citations': [1, 3]},
                {'document_id': 2, 'summary': 'Covers refunds.', 'citations': [2]},
            ]})}}
        model.post = post
        answer = model.answer('give a short summary', [
            {'id': 1, 'source': 'northstar.txt', 'text': 'Support.'},
            {'id': 2, 'source': 'acme.md', 'text': 'Refunds.'},
            {'id': 3, 'source': 'northstar.txt', 'text': 'Loyalty.'},
        ])
        prompt = captured['messages'][1]['content']
        self.assertLess(prompt.index('Excerpt [3]'), prompt.index('Document 2:'))
        self.assertEqual(answer.count('**northstar.txt**'), 1)
        self.assertEqual(answer.count('**acme.md**'), 1)
        self.assertIn('[1] [3]', answer)
        self.assertIn('35 words', captured['messages'][0]['content'])

    def test_invalid_overview_outputs_are_rejected(self):
        sources = [{'id': 1, 'source': 'a.txt', 'text': 'Support.'},
                   {'id': 2, 'source': 'b.txt', 'text': 'Refunds.'}]
        valid = [{'document_id': 1, 'summary': 'Support.', 'citations': [1]},
                 {'document_id': 2, 'summary': 'Refunds.', 'citations': [2]}]
        bad_outputs = [
            [valid[0], valid[0]],
            [dict(valid[0], citations=[2]), valid[1]],
            [dict(valid[0], citations=[]), valid[1]],
            [dict(valid[0], summary='word ' * 36), valid[1]],
        ]
        for output in bad_outputs:
            with self.subTest(output=output):
                model = Ollama()
                model.post = lambda *args: {'message': {'content': json.dumps({'summaries': output})}}
                with self.assertRaisesRegex(RuntimeError, 'invalid document summary'):
                    model.answer('Give a short summary', sources)

    def test_sample_overview_includes_end_of_document(self):
        root = Path(__file__).resolve().parents[2]
        self.agent.ingest(root / 'sample-upload.txt')
        self.agent.ingest(root / 'rag-agent' / 'docs' / 'example.md')
        result = self.agent.ask('what does the docs have , give a short summary.')
        self.assertEqual(len({s['source'] for s in result['sources']}), 2)
        context = '\n'.join(s['text'] for s in result['sources'])
        self.assertIn('LOYALTY POINTS', context)
        self.assertIn('PRODUCT FACTS', context)
        self.assertIn('costs INR 199', context)


if __name__ == "__main__":
    unittest.main()
