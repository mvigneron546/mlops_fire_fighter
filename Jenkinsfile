pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scmGit(branches: [[name: '*/jenkins-mlflow-docker-enhancement']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/MLOpsGDA/mlops_fire_fighter.git']])
            }
        }
        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
                script {
                    env.PYTHONPATH = "${env.WORKSPACE}:${env.PYTHONPATH}"
                }
                sh 'python3 -m utils.tests.test_helpers'
                sh 'python3 -m utils.tests.test_model_pipeline'
                sh 'python3 -m utils.tests.test_train_predict_model'
                sh 'python3 -m utils.tests.test_data_collection'
            }
        }
        stage('Test') {
            steps {
                sh 'python3 -m pytest'
            }
        }
    }
}

