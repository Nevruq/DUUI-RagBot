export ANNOTATOR_NAME=duui-annotator-{{model_name}}
export ANNOTATOR_VERSION=0.3.0

# Models details: anpassen mit Übergabe 
export MODEL_NAME={{model_name}}

docker build \
  --build-arg ANNOTATOR_NAME \
  --build-arg ANNOTATOR_VERSION \
  --build-arg LOG_LEVEL \
  -t ${DOCKER_REGISTRY}${ANNOTATOR_NAME}${ANNOTATOR_VERSION} \
  -f "./Dockerfile"$ \
  .

docker tag \
  ${ANNOTATOR_NAME}${ANNOTATOR_VERSION} \
  ${ANNOTATOR_NAME}:latest