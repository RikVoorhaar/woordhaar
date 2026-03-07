"""Dictionary providers and lemmatizer."""

from .base import DictionaryEntry, DictionaryProvider
from .lemmatizer import lemmatize
from .sqlite_provider import SqliteDictionaryProvider

__all__ = [
    "DictionaryEntry",
    "DictionaryProvider",
    "SqliteDictionaryProvider",
    "lemmatize",
]
