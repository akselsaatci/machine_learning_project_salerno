from client import Client, JOINTS, DEFAULT_PORT
import sys
import math

TABLE_HEIGHT = 1.0
TABLE_THICKNESS = 0.08
TABLE_LENGTH = 2.4
TABLE_WIDTH = 1.4


def run(cli, nn_wrapper):
    actor = AutoPlayerInterface(nn_wrapper)
    while True:
        state = cli.get_state()
        j = actor.update(state)
        cli.send_joints(j)


def main():
    name = 'Example Client'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)


if __name__ == '__main__':
    '''
    python client_ai.py name port host
    Default parameters:
     name: 'Example Client'
     port: client.DEFAULT_PORT
     host: 'localhost'

    To run the one simulation on the server, run this in 3 separate command shells:
    > python client_ai.py player_A
    > python client_ai.py player_B
    > python server.py

    To run a second simulation, select a different PORT on the server:
    > python client_ai.py player_A 9544
    > python client_ai.py player_B 9544
    > python server.py -port 9544    
    '''

    main()
