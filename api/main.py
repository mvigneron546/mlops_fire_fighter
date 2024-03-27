from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import pickle
import pandas as pd
import os
import yaml
import json
from datetime import date, datetime
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv


load_dotenv()

with open("../config.yml", "r") as f:
    config = yaml.safe_load(f)

ACCOUNT_URL = config['data']['account_url']
CONTAINER_NAME = config['data']['azure_container_name']
API_REQUEST_PATH = config["api"]["requests"]
API_RESPONSE_PATH = config["api"]["responses"]
API_DIRECTORY = config['data']['api_directory']
VAULT_URL = config['api']['key_vault_url']
CONNECTION_STRING = os.getenv("STORAGE_ACCOUNT_CONNECTION_STRING")

blob_service_client = BlobServiceClient\
    .from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

credentials = DefaultAzureCredential()
sc = SecretClient(vault_url=VAULT_URL, credential=credentials)

# username:password dictionary
USERS = {
    "dominik": None,
    "sevag": None,
    "marc": None
}
for user in USERS:
    USERS[user] = sc.get_secret(user).value

# set the date to track the models via date
today = date.today().strftime('%Y-%m-%d')

# Define the FastAPI app
app = FastAPI(title='MLOpsGDA API',
              description='MLOpsGDA API for the firefighter project',
              version='1.0.1',
              openapi_tags=[
                  {
                      'name': 'Home',
                      'description': 'API functionality'
                  },
                  {
                      'name': 'Predictions',
                      'description': 'Get predicted response time'
                  }
              ])

security = HTTPBasic()


# Load the pipeline model
# with open(f'./api/best_model_{today}.pkl', 'rb') as pickle_in:
#    pipe = pickle.load(pickle_in)

file_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),
                                         'data'))
with open(file_path + f'/best_model_{today}.pkl', 'rb') as pickle_in:
    pipe = pickle.load(pickle_in)


# Define the request payload model using Pydantic
class PredictionPayload(BaseModel):
    HourOfCall: int
    IncidentGroup: str
    StopCodeDescription: str
    PropertyCategory: str
    PropertyType: str
    AddressQualifier: str
    ProperCase: str
    IncGeo_WardName: str
    NumStationsWithPumpsAttending: float
    NumPumpsAttending: float
    NumCalls: float
    DateOfCall_Month: int
    PartOfDay: str

# Authentication function


def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = USERS.get(credentials.username)
    if user is None or user != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

# Expose the prediction functionality


@app.get('/', tags=['Home'])
def index():
    return {'message': 'API is working properly!'}

# Define the prediction endpoint with authentication


@app.post('/predict', tags=['Predictions'])
def predict(payload: PredictionPayload,
            user: str = Depends(authenticate_user)):
    # Convert the payload to a DataFrame
    df = pd.DataFrame([payload.dict()])

    now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    container_client.upload_blob(f"{API_REQUEST_PATH}request_{now}.json",
                                 json.dumps(payload.dict()),
                                 overwrite=True)

    # Make predictions using the pipeline model
    predictions = pipe.predict(df)
    container_client.upload_blob(f"{API_RESPONSE_PATH}response_{now}.json",
                                 json.dumps({'Predicted Response Time is':
                                             predictions.tolist()}),
                                 overwrite=True)

    api_object = {
        'request': payload.dict(),
        'response': {'Predicted Response Time is': predictions.tolist()},
    }
    container_client.upload_blob(f"{API_DIRECTORY}input_output_{now}.json",
                                 json.dumps(api_object),
                                 overwrite=True)
    # Return the predictions as a response
    return {'Predicted Response Time is': predictions.tolist()}
