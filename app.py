import os
import io
import requests 
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from datetime import datetime
from supabase import create_client, Client
from ai_engine import analyze_spine_image
from models import db, User, Image
from datetime import datetime, timedelta

app = Flask(__name__)
@app.template_filter('date_tr')
def date_tr_filter(dt):
    """
    Tarihe 3 saat ekler ve string olarak formatlar.
    Eğer tarih yoksa boş döner (Hata vermez).
    """
    if dt is None:
        return ""
    # 3 saat ekle ve formatla
    return (dt + timedelta(hours=3)).strftime('%d.%m.%Y %H:%M')

# Hesaplanan skolyoz açısına göre sınıflandırma yapan fonksiyon

# app.py içine eklenecek yeni fonksiyon

def get_diagnosis_data(angle_str):
    """Hem teşhis metnini hem de renk kodunu döndürür"""
    try:
        angle = float(angle_str)
    except:
        return "Tanımsız", "#95a5a6" # Gri

    if angle < 10:
        return "Normal (Skolyoz Yok)", "#27ae60" # Yeşil
    elif 10 <= angle < 25:
        return "Hafif Skolyoz", "#f1c40f" # Sarı
    elif 25 <= angle < 40:
        return "Orta Derece Skolyoz", "#e67e22" # Turuncu
    elif 40 <= angle < 80:
        return "Şiddetli Skolyoz", "#c0392b" # Kırmızı
    else:
        return "Çok Şiddetli Skolyoz", "#8e44ad" # Mor

# --- AYARLAR ---
app.config['SECRET_KEY'] = 'key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.icpkpxgzcagqkzoirsgw:F+7?XR*yGD/ZQaU@aws-1-eu-central-1.pooler.supabase.com:6543/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

SUPABASE_URL = "https://icpkpxgzcagqkzoirsgw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImljcGtweGd6Y2FncWt6b2lyc2d3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ0OTY5MjcsImV4cCI6MjA4MDA3MjkyN30.GnRHTQfYwvssYVfl0oEUc520yjP0aachF1zmdwwWyAg"
SUPABASE_BUCKET = "images"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ROTALAR ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')             
        doctor_code = request.form.get('doctor_code')

        # DÜZELTME BURADA: 'doktor' yerine 'doctor' yazmalısın
        if role == 'doctor' and doctor_code != 'MED123':
            flash('Hatalı Doktor Kodu!', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        # ... kodun geri kalanı aynı
        new_user = User(username=username, password=hashed_password, role=role)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Kayıt başarılı!', 'success')
            return redirect(url_for('login'))
        except:
            flash('Kullanıcı adı alınmış.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('upload_file'))
        else:
            flash('Hatalı giriş.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files: return render_template('index.html', error="Dosya yok.")
        file = request.files['file']
        
        patient_name = current_user.username if current_user.role == 'hasta' else request.form.get('patient_name')
        if not patient_name: return render_template('index.html', error="Hasta adı giriniz.")

        if file:
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                
                supabase.storage.from_(SUPABASE_BUCKET).upload(unique_filename, file.read(), {"content-type": file.content_type})
                image_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)

                # Yeni resim kaydederken:
                new_image = Image(filename=image_url, user_id=str(current_user.id), patient_name=patient_name)
                db.session.add(new_image)
                db.session.commit()
                return redirect(url_for('process_image', image_id=new_image.id))
            except Exception as e:
                flash(f"Hata: {e}", "error")
    return render_template('index.html', name=current_user.username, role=current_user.role)

@app.route('/dashboard')
@login_required
def dashboard():
    search = request.args.get('search')
    # DÜZELTME: current_user.id'yi str() içine aldık
    query = Image.query.filter_by(user_id=str(current_user.id)) 
    if search: query = query.filter(Image.patient_name.ilike(f"%{search}%"))
    return render_template('dashboard.html', images=query.order_by(Image.upload_date.desc()).all(), name=current_user.username, role=current_user.role)

# app.py dosyasındaki process_image fonksiyonunu şöyle güncelle:

@app.route('/process/<int:image_id>')
@login_required
def process_image(image_id):
    try:
        img_rec = Image.query.get_or_404(image_id)
        # Başkasının resmine erişimi engelleme kontrolünde:
        if str(img_rec.user_id) != str(current_user.id): 
            return redirect(url_for('dashboard'))

        resp = requests.get(img_rec.filename)
        if resp.status_code != 200: return render_template('process.html', error="Resim indirilemedi.")

        # AI Motorundan sonuçları al (Önceki düzenlemeye göre 4 değer dönüyor olabilir, dikkat et)
        # Eğer 3 değer dönüyorsa: processed, preprocessed, angle = ...
        # Eğer 4 değer dönüyorsa: processed, preprocessed, angle, mask = ...
        # Kodun hata vermemesi için dönen değerleri bir değişkene atayıp içinden alalım:
        
        # AI Sonuçlarını al
        results = analyze_spine_image(io.BytesIO(resp.content))
        processed_base64 = results[0]
        preprocessed_base64 = results[1]
        cobb_angle = results[2]
        mask_image = results[3] if len(results) > 3 else None #eğer maske varsa

        # Fonksiyondan hem METİN hem RENK alıyoruz
        diag_text, diag_color = get_diagnosis_data(cobb_angle)

        return render_template(
        'process.html',
        filename=img_rec.filename,
        processed_image_data=processed_base64,
        preprocessed_image=preprocessed_base64,
        mask_image=mask_image,
        cobb_angle=cobb_angle,
        
        # HTML'e iki ayrı değişken gönderiyoruz
        diagnosis=diag_text,      
        diagnosis_color=diag_color, 
        
        name=current_user.username,
        role=current_user.role
    )
    except Exception as e:
        return render_template('process.html', error=f"Hata: {e}")

# --- MOBİL UYGULAMA İÇİN REST API UÇ NOKTASI ---

@app.route('/api/analyze', methods=['POST'])
def api_analyze_image():
    # 1. İstekte dosya olup olmadığını kontrol et
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "İstekte 'file' adında bir dosya bulunamadı."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "Herhangi bir dosya seçilmedi."}), 400

    try:
        # 2. Dosyayı belleğe al ve yapay zeka motoruna gönder
        file_bytes = io.BytesIO(file.read())
        results = analyze_spine_image(file_bytes)
        
        # 3. Sonuçları ayrıştır (Mevcut process_image fonksiyonundaki gibi)
        processed_base64 = results[0]
        preprocessed_base64 = results[1]
        cobb_angle = results[2]
        # Eğer maske dönüyorsa onu da al
        mask_image = results[3] if len(results) > 3 else None 

        # 4. Teşhis ve renk verilerini hesapla
        diag_text, diag_color = get_diagnosis_data(cobb_angle)

        # 5. Mobil uygulamaya JSON formatında yanıt dön
        return jsonify({
            "success": True,
            "data": {
                "cobb_angle": cobb_angle,
                "diagnosis": diag_text,
                "diagnosis_color": diag_color,
                "processed_image": processed_base64,
                "preprocessed_image": preprocessed_base64,
                "mask_image": mask_image
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Analiz sırasında hata oluştu: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    # host='0.0.0.0' sayesinde telefonun aynı Wi-Fi üzerinden buraya erişebilir
    app.run(host='0.0.0.0', port=5000, debug=True)