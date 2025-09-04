# src/api/middleware.py

import json
import re
from fastapi import Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import HTTPException
from starlette.requests import Request as StarletteRequest


# --- CASE CONVERSION HELPERS ---
def to_camel_case(snake_str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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


# --- CUSTOM REQUEST CLASS ---
class ModifiedRequest(StarletteRequest):
    def __init__(self, scope, receive, send, body: bytes):
        super().__init__(scope, receive, send)
        self._body = body
        self._body_consumed = False
    
    async def body(self) -> bytes:
        if not self._body_consumed:
            self._body_consumed = True
        return self._body


# --- MIDDLEWARE CLASS ---
class CaseConversionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-type") == "application/json":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)
                    request_body_snake = convert_dict_keys(data, to_snake_case)
                    
                    # Create a new body with converted keys
                    new_body = json.dumps(request_body_snake).encode()
                    
                    # Create a modified request with the new body
                    modified_request = ModifiedRequest(
                        scope=request.scope,
                        receive=request.receive,
                        send=request._send,
                        body=new_body
                    )
                    
                    # Use the modified request for the next middleware/endpoint
                    response = await call_next(modified_request)
                    
                    # Handle response conversion
                    if response.headers.get("content-type") == "application/json":
                        response_body = b""
                        async for chunk in response.body_iterator:
                            response_body += chunk

                        try:
                            data = json.loads(response_body.decode())
                            response_body_camel = convert_dict_keys(data, to_camel_case)

                            # Create new response and let FastAPI handle Content-Length
                            headers = dict(response.headers)
                            # Remove Content-Length to let JSONResponse calculate it correctly
                            headers.pop('content-length', None)
                            
                            return JSONResponse(
                                content=response_body_camel, 
                                status_code=response.status_code, 
                                headers=headers
                            )
                        except json.JSONDecodeError:
                            # If response is not valid JSON, return it as is
                            # Let Response handle Content-Length automatically
                            headers = dict(response.headers)
                            headers.pop('content-length', None)
                            return Response(
                                content=response_body, 
                                status_code=response.status_code, 
                                headers=headers
                            )
                    
                    return response
                    
            except json.JSONDecodeError:
                # If body is not valid JSON, proceed without conversion
                pass

        # For non-JSON requests or when conversion fails, proceed normally
        response = await call_next(request)

        # Convert response to camelCase for all JSON responses
        if response.headers.get("content-type") == "application/json":
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            try:
                data = json.loads(response_body.decode())
                response_body_camel = convert_dict_keys(data, to_camel_case)

                # Create new response and let FastAPI handle Content-Length
                headers = dict(response.headers)
                # Remove Content-Length to let JSONResponse calculate it correctly
                headers.pop('content-length', None)
                
                return JSONResponse(
                    content=response_body_camel, 
                    status_code=response.status_code, 
                    headers=headers
                )
            except json.JSONDecodeError:
                # If response is not valid JSON, return it as is
                # Let Response handle Content-Length automatically
                headers = dict(response.headers)
                headers.pop('content-length', None)
                return Response(
                    content=response_body, 
                    status_code=response.status_code, 
                    headers=headers
                )

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
        content={
            "message": "An unexpected internal server error occurred.",
            "code": "INTERNAL_SERVER_ERROR",
        },
    )