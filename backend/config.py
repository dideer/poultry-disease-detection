# config.py
# PostgreSQL database credentials & connection

import os

import os

# PostgreSQL Database Configuration
DB_NAME = os.getenv('DB_NAME', 'chicken_db')        # Database name
DB_USER = os.getenv('DB_USER', 'chicken_user')      # User we created
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Ishimwe123')# Password for chicken_user
DB_HOST = os.getenv('DB_HOST', 'localhost')         # Usually localhost
DB_PORT = os.getenv('DB_PORT', '5432')              # Default PostgreSQL port