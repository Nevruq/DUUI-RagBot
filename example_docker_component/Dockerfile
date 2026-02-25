FROM python:3.10
WORKDIR /usr/src/app
EXPOSE 9714
COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='debajyotimaz/codemix_hate')"
# service script
COPY ./src/main/python/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/hatechecker.py ./hatechecker.py
COPY ./src/main/python/duui_hate.lua ./duui_hate.lua
COPY ./src/main/python/duui_hate.py ./duui_hate.py

ENTRYPOINT ["uvicorn", "duui_hate:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

