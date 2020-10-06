import os
import psycopg2
import time
import json
from sshtunnel import SSHTunnelForwarder

def get_connection_hardcode():
 # dir_path = os.path.dirname(os.path.realpath(__file__))
    #file_key = os.path.join(dir_path, 'wise.pem')
    file_key = 'wise.pem'
    result = None
    server = SSHTunnelForwarder(
        ("52.14.136.154", 22),
        ssh_username="ec2-user",
        ssh_private_key="wise.pem",
        remote_bind_address=("social-automation.cqt472ritvtr.us-east-2.rds.amazonaws.com", 5432),
        local_bind_address=('127.0.0.1', 5432),
    )
    server.start()
    conn = psycopg2.connect(
    host='127.0.0.1',
    port=5432,
    user="postgres",
    password="fwQDYEq8S9zwEcNj1PkU",
    dbname="social")

    return server, conn


def execute(query, params):
    server, conn = get_connection_hardcode()
    cur = conn.cursor()
    cur.execute(query, params)
    result = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    server.close()

    return result


def insert_tweets(tweet_data):
    with open(tweet_data, 'r') as file:
        data = json.load(file)

    for tweet in data:
        content = tweet['text']
        network = "twitter"
        status = 0
        creation_date_utc = int(time.time())
        content_meta = json.dumps(tweet)
        item_id = creation_date_utc + 1
        query =  """
                    INSERT INTO content_items(content, network, status, campaign_id, content_meta, item_id)
                    VALUES(%s, %s, %s, %s, %s, %s)
                    RETURNING item_id"""

        params = (content, network, status, 0, content_meta, item_id)

        result = execute(query, params)
        print("added to database with id {}".format(result))
