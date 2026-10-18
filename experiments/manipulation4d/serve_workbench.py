"""Attach the experimental API without changing the existing inference code digest."""
from greenwich_pipelines.viewer.server import create_app
from workbench_api import create_api
import uvicorn

app=create_app('/opt/arenalab/configs/viewer.cloud.json')
app.mount('/api/4d',create_api())
from dataset_io.routes import protect_uploaded_assets
protect_uploaded_assets(app)
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=8765)
