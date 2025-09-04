# src/api/middleware.py

import json
import re
from fastapi import Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import HTTPException

# --- CASE CONVERSION HELPERS ---
def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake_case(camel_str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_dict_keys(d, convert_func):
    if not isinstance(d, dict):
        return d
    
    new_dict = {}
    for k, v in d.items():
        new_key = convert_func(k)
        if isinstance(v, dict):
            new_dict[new_key] = convert_dict_keys(v, convert_func)
        elif isinstance(v, list):
            new_dict[new_key] = [convert_dict_keys(i, convert_func) for i in v]
        else:
            new_dict[new_key] = v
    return new_dict

# --- MIDDLEWARE CLASS ---
class CaseConversionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-type") == "application/json":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)
                    request_body_snake = convert_dict_keys(data, to_snake_case)
                    
                    # Modify the request scope to replace the body
                    async def receive():
                        return {"type": "http.request", "body": json.dumps(request_body_snake).encode()}
                    request._receive = receive
            except json.JSONDecodeError:
                # If body is not valid JSON, proceed without conversion
                pass

        response = await call_next(request)

        # Convert response body from snake_case to camelCase if it's a JSON response
        if "application/json" in response.headers.get("content-type", ""):
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            try:
                data = json.loads(response_body.decode())
                response_body_camel = convert_dict_keys(data, to_camel_case)
                
                # Return a new JSONResponse. FastAPI will automatically handle
                # calculating the correct Content-Length for the new body.
                return JSONResponse(content=response_body_camel, status_code=response.status_code)
            
            except json.JSONDecodeError:
                # If response is not valid JSON, we still need to rebuild it
                # to avoid "response already consumed" errors.
                return Response(content=response_body, status_code=response.status_code, headers=response.headers)
        
        return response

# --- ERROR HANDLERS ---
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "code": f"HTTP_{exc.status_code}_ERROR"},
    )

async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An unexpected internal server error occurred.", "code": "INTERNAL_SERVER_ERROR"},
    )