# Export Satellite Dataset to your Drive

## Installations
```bash
pip install earthengine-api
```
then
```bash
earthengine authenticate 
```

## Preparation
if you want to export to drive directly using a colab notebook
```python
from google.colab import drive
drive.mount('/content/drive')

from google.colab import auth
auth.authenticate_user()
```
