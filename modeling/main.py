import os
from datetime import date
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()

try:
    from .utils_modeling.model_pipeline import get_pipeline
    from .utils_modeling.train_predict_model import train_model, predict_model
except ImportError:
    from utils_modeling.model_pipeline import get_pipeline
    from utils_modeling.train_predict_model import train_model, predict_model

with open("../config.yml", "r") as f:
    config = yaml.safe_load(f)

CONTAINER_NAME = config['data']['azure_container_name']
CONNECTION_STRING = os.getenv("STORAGE_ACCOUNT_CONNECTION_STRING")
TRAIN_PATH = config['data']['train_newest']
TEST_PATH = config['data']['test_newest']

blob_service_client = BlobServiceClient\
    .from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# set the date to track the models via date
today = date.today().strftime('%Y-%m-%d')

# path = os.path.abspath(os.path.join(os.getcwd(), 'modeling/mlruns'))
path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'modeling/mlruns'))

mlflow.set_tracking_uri('file://' + path)

experiment_name = 'FireFighter'

# Check if the experiment exists; if not, create it
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

experiment_id = experiment.experiment_id

file_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'data'))

# Read Data
train = pd.read_parquet(BytesIO(
        container_client.download_blob(TRAIN_PATH).readall()))
test = pd.read_parquet(BytesIO(
        container_client.download_blob(TEST_PATH).readall()))

# Feature sets
# The lists below contain the features available in the data.
# target_feature: the target feature
# drop_features: features that will be dropped and are not considered
# cat_features: the categorical features
# num_features: the numerical features
# id_feature: the id feature
target = 'FirstPumpArriving_AttendanceTime_min'

drop_features = ['DateOfCall', 'CalYear', 'TimeOfCall', 'SpecialServiceType',
                 'Postcode_full', 'Postcode_district', 'UPRN', 'USRN',
                 'IncGeo_BoroughName', 'IncGeo_WardCode', 'IncGeo_WardNameNew',
                 'Easting_m', 'Northing_m', 'Easting_rounded',
                 'Northing_rounded', 'Latitude', 'Longitude', 'FRS',
                 'IncidentStationGround', 'FirstPumpArriving_AttendanceTime',
                 'SecondPumpArriving_AttendanceTime',
                 'SecondPumpArriving_DeployedFromStation', 'PumpCount',
                 'PumpHoursRoundUp', 'FirstPumpArriving_DeployedFromStation',
                 'IncGeo_BoroughCode', 'Notional Cost (£)']

cat_features = ['IncidentGroup', 'StopCodeDescription', 'PropertyCategory',
                'PropertyType', 'AddressQualifier', 'ProperCase',
                'IncGeo_WardName', 'PartOfDay']

num_features = ['HourOfCall', 'NumStationsWithPumpsAttending',
                'NumPumpsAttending', 'NumCalls', 'DateOfCall_Month',
                ]

id_feature = 'IncidentNumber'

full_list_features = set(cat_features + num_features)

# Drop features & define modeling data sets
train = train.drop(columns=drop_features, axis=1)
test = test.drop(columns=drop_features, axis=1)

X_train = train.drop(columns=[target]+[id_feature])
y_train = train[target]
X_test = test.drop(columns=[target]+[id_feature])
y_test = test[target]

assert full_list_features == set(X_train.columns), 'missmatch in sets'
assert full_list_features == set(X_test.columns), 'missmatch in sets'

# Define algorithm options
algorithms = ['lgbm', 'xgboost', 'rf']
best_rmse = float('inf')
best_algorithm = None
best_predictions = None

# Try each algorithm and choose the one with the lowest RMSE
for algorithm in algorithms:
    pipe = get_pipeline(
        num_features=num_features,
        cat_features=cat_features,
        algorithm=algorithm,
        use_grid_search=False
    )

    model = train_model(
        pipeline=pipe,
        xtrain=X_train,
        ytrain=y_train
    )

    evaluation = predict_model(
        trained_model=model,
        xtest=X_test,
        ytest=y_test
    )

    rmse = evaluation['Root Mean Squared Error']
    if rmse < best_rmse:
        best_rmse = rmse
        best_algorithm = algorithm
        best_predictions = evaluation

# Save the best model
best_model = train_model(
    pipeline=get_pipeline(
        num_features=num_features,
        cat_features=cat_features,
        algorithm=best_algorithm,
        use_grid_search=False
    ),
    xtrain=X_train,
    ytrain=y_train
)

# file_path = os.path.abspath(os.path.join(
# os.path.dirname(os.getcwd()), 'data'))

pickle_out = open(file_path + f'/best_model_{today}.pkl', 'wb')
pickle.dump(best_model, pickle_out)
pickle_out.close()

# Loading the metrics and MLflow parameters
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param('max_depth', best_model._final_estimator.max_depth)
    mlflow.log_param('num_leaves', best_model._final_estimator.num_leaves)
    mlflow.log_param('n_estimators', best_model._final_estimator.n_estimators)
    mlflow.log_param('best_algorithm', best_algorithm)

    mlflow.log_metric('rmse', evaluation.get('Root Mean Squared Error'))
    mlflow.log_metric('mae', evaluation.get('Mean Absolute Error'))
    mlflow.log_metric('mse', evaluation.get('Mean Squared Error'))

    mlflow.sklearn.log_model(best_model, 'model')
