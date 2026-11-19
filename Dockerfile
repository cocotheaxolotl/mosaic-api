FROM python:3.11-slim

WORKDIR /app

# Install system fonts for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints-production.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints-production.txt
RUN python -m pip check

COPY . .

# Fail the build before rollout if imports or credit regressions fail.
RUN python -c "import app; import annual_credits" \
    && python -m unittest -q test_annual_credits test_billing_lifecycle

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
