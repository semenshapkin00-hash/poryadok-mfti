"""Порядок — полностью локальный навигатор регламентов с точными цитатами."""
from .models import DocumentInput
from .parsing import ingest
from .core import answer
__all__=['DocumentInput','ingest','answer']
