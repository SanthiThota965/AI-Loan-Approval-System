import sqlite3

# Connect database
conn = sqlite3.connect("loan_applications.db")

cursor = conn.cursor()

# Create table

cursor.execute("""

CREATE TABLE IF NOT EXISTS applications (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    customer_name TEXT,

    email TEXT,

    gender TEXT,

    married TEXT,

    education TEXT,

    applicant_income REAL,

    coapplicant_income REAL,

    loan_amount REAL,

    loan_term REAL,

    credit_history REAL,

    property_area TEXT,

    prediction TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

""")

conn.commit()

conn.close()

print("Database and table created successfully")