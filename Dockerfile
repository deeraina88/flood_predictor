# pull python base image
FROM python:3.10-alpine
# copy application files
ADD /flood_predictor_model_api /flood_predictor_model_api/
# specify working directory
WORKDIR /flood_predictor_model_api
# update pip
RUN pip install --upgrade pip
# install dependencies
RUN pip install -r requirements.txt
# expose port for application
EXPOSE 8001
# start fastapi application
CMD ["python", "app/main.py"]