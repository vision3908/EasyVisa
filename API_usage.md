```markdown
# Visa Approval Prediction API - Usage Guide

## Quick Start

### Option 1: Run with Docker (Recommended)
```bash
docker run -d -p 8000:8000 visionmlops/visa-api:latest
```

### Option 2: Run Locally
```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
**Endpoint**: `GET /`

**Description**: Check if API is running

**Example**:
```bash
curl http://localhost:8000/
```

**Response**:
```json
{
  "message": "Visa Approval Prediction API",
  "status": "running",
  "version": "1.0.0"
}
```

### 2. Predict Visa Approval
**Endpoint**: `POST /predict`

**Description**: Predict visa approval for a given application

**Request Body**:
```json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": 500,
  "yr_of_estab": 2010,
  "region_of_employment": "West",
  "prevailing_wage": 85000.0,
  "unit_of_wage": "Yearly",
  "full_time_position": "Y"
}
```

**Example with curl**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "continent": "Asia",
    "education_of_employee": "Master'\''s",
    "has_job_experience": "Y",
    "requires_job_training": "N",
    "no_of_employees": 500,
    "yr_of_estab": 2010,
    "region_of_employment": "West",
    "prevailing_wage": 85000.0,
    "unit_of_wage": "Yearly",
    "full_time_position": "Y"
  }'
```

**Response**:
```json
{
  "prediction": "Certified",
  "probability_certified": 0.92,
  "probability_denied": 0.08
}
```

### 3. Get Model Information
**Endpoint**: `GET /model-info`

**Description**: Get information about the ML model

**Example**:
```bash
curl http://localhost:8000/model-info
```

## Input Fields

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| continent | string | Applicant's continent | Asia, Europe, Africa, North America, South America, Oceania |
| education_of_employee | string | Education level | High School, Bachelor's, Master's, Doctorate |
| has_job_experience | string | Has prior work experience | Y, N |
| requires_job_training | string | Requires job training | Y, N |
| no_of_employees | integer | Company size | Positive integer |
| yr_of_estab | integer | Year company was established | 1900-2024 |
| region_of_employment | string | US region | Northeast, South, Midwest, West, Island |
| prevailing_wage | float | Wage offered | Positive number |
| unit_of_wage | string | Wage unit | Yearly, Monthly, Weekly, Hourly |
| full_time_position | string | Full-time position | Y, N |

## Interactive Documentation

Visit http://localhost:8000/docs for interactive Swagger UI where you can:
- Try all endpoints
- See request/response schemas
- Test with example data

## Error Handling

If prediction fails:
```json
{
  "error": "Error message details",
  "message": "Prediction failed"
}
```

## Production Considerations

- **Rate Limiting**: Not implemented (add nginx/API gateway for production)
- **Authentication**: Not implemented (add OAuth2/JWT for production)
- **Logging**: Basic logging to stdout (use centralized logging for production)
- **Monitoring**: Add Prometheus metrics for production

## Performance

- **Latency**: ~50ms per prediction (single request)
- **Throughput**: ~100 requests/second (single instance)
- **Memory**: ~200MB RAM per instance
```
