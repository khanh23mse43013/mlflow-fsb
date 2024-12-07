from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

    
from src import controllers

app = FastAPI(
    title="MLOps",
    version="1.0.0",
)
app.include_router(controllers.predict_controller)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title = "MLOps",
        version = "1.0.0",
        description = "MLOps API",
        routes = app.routes,
    )  
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi