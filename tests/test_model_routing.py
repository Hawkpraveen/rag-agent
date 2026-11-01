import unittest
from unittest.mock import patch

from model_routing import select_route
from rag import Ollama
from main import QuestionRequest
from pydantic import ValidationError


class RoutingTests(unittest.TestCase):
    models = {'fast': 'small', 'thorough': 'large', 'code': 'coder', 'default': 'small'}

    def route(self, question, sources=None, mode='auto', installed=None):
        return select_route(question, sources or [], mode, self.models,
                            installed if installed is not None else ['small', 'large', 'coder'])

    def test_task_and_context_selection(self):
        self.assertEqual(self.route('Give a short summary')['model'], 'small')
        self.assertEqual(self.route('Compare return policies')['model'], 'large')
        self.assertEqual(self.route('Debug this Python function')['model'], 'coder')
        self.assertEqual(self.route('Explain', [{'text': 'a' * 6501}])['model'], 'large')
        self.assertEqual(self.route('Explain', [{'source': str(i)} for i in range(4)])['model'], 'large')

    def test_overrides_and_missing_models(self):
        self.assertEqual(self.route('Compare policies', mode='fast')['model'], 'small')
        self.assertEqual(self.route('Support hours?', mode='thorough')['model'], 'large')
        result = self.route('Compare policies', installed=['small:latest'])
        self.assertEqual(result['model'], 'small:latest')
        self.assertIn('unavailable', result['reason'])
        with self.assertRaises(RuntimeError):
            self.route('Hello', installed=['embedding-only'])
        with self.assertRaises(ValueError):
            self.route('Hello', mode='arbitrary-model')
        with self.assertRaises(ValidationError):
            QuestionRequest(question='Hello', model_mode='arbitrary-model')

    def test_auto_retries_validation_failure_and_retains_usage(self):
        model = Ollama()
        model.routing_models = self.models
        model.installed_models = lambda: ['small', 'large']
        calls = []
        def post(endpoint, payload):
            calls.append(payload['model'])
            return {'model': payload['model'], 'message': {'content':
                    'Wrong [9]' if payload['model'] == 'small' else 'Support is open [1].'}}
        model.post = post
        answer, routing = model.answer_routed('Support hours?', [{'id': 1, 'text': 'Open'}])
        self.assertEqual(calls, ['small', 'large'])
        self.assertEqual(routing['initial_model'], 'small')
        self.assertEqual(routing['model'], 'large')
        self.assertIn('[1]', answer)
        self.assertEqual([u['model_name'] for u in model.usage], ['small', 'large'])

    def test_explicit_mode_does_not_escalate_and_network_errors_do_not_retry(self):
        for mode, error in [('fast', 'Invalid citation'), ('auto', 'Cannot reach Ollama')]:
            model = Ollama()
            model.routing_models = self.models
            model.installed_models = lambda: ['small', 'large']
            with patch.object(model, 'answer', side_effect=RuntimeError(error)) as answer:
                with self.assertRaises(RuntimeError):
                    model.answer_routed('Support hours?', [], mode)
                self.assertEqual(answer.call_count, 1)
