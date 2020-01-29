import psycopg2

def get_db_connection():
   psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
   conn = psycopg2.connect(host='10.107.24.87', port='5432', user='dylan',
                           password='ch4ng3meplz', dbname='brexit')
   return conn

if __name__ == "__main__":
    print("creating connection")
    conn = get_db_connection()
    print("connection succesful")
    cursor = conn.cursor()
    print("querying databse")
    cursor.execute("select screen_name_to, count(*) from retweet group by screen_name_to")
    results = cursor.fetchall()
    cursor.execute("select * from tweet limit 10")
    print("query sucessful printing outputs")
    for i in cursor.fetchall():
        print(i)
    conn.close()
