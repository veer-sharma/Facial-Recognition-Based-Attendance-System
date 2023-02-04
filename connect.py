import psycopg2
import yaml

db = yaml.safe_load(open('static/db.yaml'))

class PostgresConnection:
    def __init__(self, host, database, user, password):
        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        self.cur = self.conn.cursor()

    def insert(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.conn.commit()

    def read(self, sql_query):
        self.cur.execute(sql_query)
        rows = self.cur.fetchall()
        return rows

    def create(self, sql_query):
        self.cur.execute(sql_query)
        self.conn.commit()

    def update(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.conn.commit()

    def delete(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

conn = PostgresConnection(db['mysql_host'], db['mysql_db'], db['mysql_user'], db['mysql_password'])
