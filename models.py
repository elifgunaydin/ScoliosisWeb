#models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.String, nullable=False) # Eğer ForeignKey varsa onu kaldırabilirsin, çünkü artık ID tipleri farklı
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # --- YENİ EKLENEN SÜTUN ---
    patient_name = db.Column(db.String(100), nullable=False, default="Bilinmiyor")