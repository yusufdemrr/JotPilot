from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse


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
