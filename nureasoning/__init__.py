"""nuReasoning devkit."""

import os

__version__ = "0.1.0"

# Hub weights default to <repo>/models/.hf unless HF_HOME is already set.
# HF_HOME also controls where huggingface_hub looks for the login token, so
# keep HF_TOKEN_PATH pointed at the user cache written by `hf auth login`.
_default_hf_home = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", ".hf")
)
_default_token_path = os.path.expanduser("~/.cache/huggingface/token")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _default_hf_home
if "HF_TOKEN_PATH" not in os.environ:
    os.environ["HF_TOKEN_PATH"] = _default_token_path

