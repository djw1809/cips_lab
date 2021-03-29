import psycopg2

def get_db_connection():
   psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
   conn = psycopg2.connect(host='artislg-dev.cqt472ritvtr.us-east-2.rds.amazonaws.com', port='5432', dbname = 'postgres', user='dylan_weber',
                           password='cvg0bf1DHC&Hvj%X%7!NmSt0WY0XlG')
   return conn


conn = get_db_connection()
curor = conn.cursor() 

def executee(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except psycopg2.Error as e:
        print(e.pgerror)
        conn.rollback()
        return e


if __name__ == "__main__":
    print("creating connection")
    conn = get_db_connection()
    cursor = conn.cursor()
    print("connection sucessful")
