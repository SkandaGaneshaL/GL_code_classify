from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_upload_ui_renders_without_calling_oci():
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    assert len(app.file_uploader) == 1
    assert any(button.label == "Extract and classify invoice" for button in app.button)
