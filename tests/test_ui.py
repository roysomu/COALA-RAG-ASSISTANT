from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_setup_screen_without_credentials(monkeypatch):
    import app
    from src.config import Settings
    monkeypatch.setattr('src.config.load_settings', lambda: Settings(_env_file=None))
    app.cached_runtime.clear()
    app.cached_pool.clear()
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=20).run()
    assert not ui.exception
    assert any('Complete local setup' in element.value for element in ui.warning)
    assert any('Privacy:' in element.value for element in ui.info)


def test_connected_tabs_and_confirmations(monkeypatch, settings):
    from unittest.mock import Mock
    from pydantic import SecretStr
    from datetime import datetime, timezone
    import app
    settings.database_url = SecretStr('postgresql://u:p@ep-test.neon.tech/db?sslmode=require')
    runtime = Mock()
    row = {'thread_id': 'test-thread', 'user_id': 'local-user', 'status': 'active', 'message_count': 0,
           'last_activity': datetime.now(timezone.utc), 'summary_saved_at': None, 'metadata': {}}
    runtime.threads.get.return_value = row
    runtime.threads.list.side_effect = lambda user_id, status=None: [row] if status == 'active' else []
    runtime.assistant.history.return_value = {}
    runtime.documents.list.return_value = []
    runtime.memories.list.return_value = []
    runtime.retention.auto_prune.return_value = None
    monkeypatch.setattr('src.config.load_settings', lambda: settings)
    monkeypatch.setattr('src.db.create_pool', lambda settings: Mock())
    monkeypatch.setattr('src.runtime.create_runtime', lambda *a: runtime)
    monkeypatch.setattr('src.storage_monitor.storage_report', lambda *a: {
        'database': {'pretty': '10 MB'}, 'warning': False, 'tables': [],
        'vector_indexes': [], 'counts': {}, 'memories': []})
    app.cached_runtime.clear()
    app.cached_pool.clear()
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=20).run()
    assert not ui.exception
    assert not ui.error
    assert [tab.label for tab in ui.tabs] == ['Chat', 'Documents', 'Memory and diagnostics']
    delete_button = next(button for button in ui.button if button.label == 'Delete current conversation')
    assert delete_button.disabled
    assert next(button for button in ui.button if button.label == 'Delete matching memories').disabled
    ui.chat_input[0].set_value('What is retention?')
    runtime.assistant.ask.return_value = {'next_thread_id': None, 'final_answer': '365 days'}
    ui.run()
    runtime.assistant.ask.assert_called_once_with('What is retention?', 'test-thread', 'local-user', 'Auto', 5)
    assert not ui.exception
    app.cached_runtime.clear()
    app.cached_pool.clear()
