import psycopg2

def get_db_connection():
   psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
   conn = psycopg2.connect(host='10.107.24.87', port='5432', user='dylan',
                           password='ch4ng3meplz', dbname='brexit')
   return conn

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
def get_retweet_edge_list(conn, number_of_edges):
    cursor = conn.cursor() 
    cursor.execute("SELECT retweet.tid, tweet.tid, tweet.uid, t2.uid FROM retweet, tweet, (SELECT * FROM tweet) t2 WHERE retweet.tid = tweet.tid AND retweet.retweet_tid = t2.tid LIMIT 10")
    results = cursor.fetchall()

    return results 



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
