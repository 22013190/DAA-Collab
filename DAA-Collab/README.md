# Create venv using Python 3.12
py -3.12 -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel
```

**Install dependencies**
```powershell
pip install -r requirements.txt
```

**How to run the agent**
- Basic run (relative path):

```powershell
python .\main.py --instruction "Give me an overview and histogram of 'value'" --file traffic_incidents_data.csv
```

- Absolute path (recommended if running from a different directory):