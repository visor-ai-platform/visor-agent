FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --upgrade pip \
	&& pip install -e /app \
	&& useradd --create-home --shell /usr/sbin/nologin visor
USER visor

EXPOSE 8000
CMD ["uvicorn", "visor_agent.agent:app", "--host", "0.0.0.0", "--port", "8000"]
