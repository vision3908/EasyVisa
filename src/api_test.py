from fastapi import FastAPI

# Create API instance
app = FastAPI(title="My First API")

# Define endpoint
@app.get("/")
def home():
    return {"message": "Hello, World!"}

@app.get("/add")
def add_numbers(a: int, b: int):
    result = a + b
    return {"result": result}
