"""Shared Flask extensions — instantiated here so models.py and app.py can share them without circular imports."""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
