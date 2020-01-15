import psycopg2
def get_db_connection():
   psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
   conn = psycopg2.connect(host='10.107.24.87', port='5432', user='dylan',
                           password='ch4ng3meplz', dbname='brexit')
   return conn
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("select screen_name_to, count(*) from retweet group by screen_name_to")
results = cursor.fetchall()
cursor.execute("select * from tweet limit 10")
for i in cursor.fetchall():
    print(i)
conn.close()
