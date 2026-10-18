
data_generator.py
$Project = "GlucoTwin_GHRaisoni"
New-Item -ItemType Directory -Force -Path "$Project\architecture", "$Project\presentation", "$Project\src", "$Project\data", "$Project\models" | Out-Null
Set-Location $Project

python -m venv venv
.\venv\Scripts\Activate.ps1

@"
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.32.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.44.0
"@ | Out-File -FilePath requirements.txt -Encoding utf8

pip install --upgrade pip
pip install -r requirements.txt

New-Item -ItemType File -Force -Path src\__init__.py, src\data_generator.py, src\model.py, src\train.py, src\app.py, README.md, LICENSE, .gitignore | Out-Null