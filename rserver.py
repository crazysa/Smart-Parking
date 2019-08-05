import MySQLdb
import socket

db=MySQLdb.connect(host='localhost',
                   user='root',
                   passwd='root123',
                   db='MStorage')
cur=db.cursor()

s=socket.socket()
port=8000
host=socket.gethostname()
s.bind((host,port))
s.listen(5)

print('Listening...')

while True:
    conn,addr=s.accept()
    print('Established connection from '+str(addr))
    data=conn.recv(1024)
    print('Recived '+str(data))
    print('Executing...')
    cur.execute('delete from NWorking')
    print("test")
    datas=data.split()
    if datas[0] == 'select':
        thing=cur.fetchone()[0]
        print(thing)
        print('Sending date...')
        conn.send(thing)
        print('Sent')
        continue
    print('Excecuted')
    connd.commit()