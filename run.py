"""Start the agent on the host/port configured in .env."""
import settings  # noqa: F401 -- load before reading configuration
import os
import uvicorn

if __name__ == '__main__':
    uvicorn.run('main:app', host=os.getenv('HOST', '127.0.0.1'),
                port=int(os.getenv('PORT', '3000')),
                reload=os.getenv('RELOAD', 'true').lower() == 'true')
