docker run -it\
 -v "$(pwd):/home/app"\
 -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI\
 -e PORT=4000\
 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID\
 -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
 -e BACKEND_STORE_URI=$BACKEND_STORE_URI\
 -e ARTIFACT_ROOT=$ARTIFACT_ROOT\
 jedha/appointment-cancellation-detector python train.py --n_estimators=100 --min_samples_split=2