import channel
import numpy as np
from neural_network_wrapper import NNWrapper

JOINTS = 11
STATE_DIMENSION = 37
DEFAULT_PORT = 9543


class Client:
    def __init__(self, name, host='localhost', port=DEFAULT_PORT):
        key = name.encode('utf8')
        self.channel = channel.ClientChannel(host, port, key)
        self.nn_wrapper = NNWrapper()

    def get_state(self, blocking=True):
        timeout = None
        if blocking:
            timeout = -1
        last_msg = self.channel.receive(timeout)
        if last_msg is None:
            return None
        msg = self.channel.receive()
        while msg is not None:
            last_msg = msg
            msg = self.channel.receive()
        state = channel.decode_float_list(last_msg)
        return np.array(state)

    def send_joints(self, joints):
        joints = list(joints)
        if len(joints) != JOINTS:
            raise ValueError('Invalid number of elements')
        msg = channel.encode_float_list(joints)
        if msg is None:
            raise ValueError('Invalid joints vector')
        self.channel.send(msg)

    def close(self):
        self.channel.close()

    def run(self):

        try:
            while True:
                state = self.get_state()
                if state is None:
                    break

                # Compute the action using NNWrapper
                action = self.nn_wrapper.update(state)

                # Send the action to the server
                self.send_joints(action)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping the client.")
        finally:
            self.close()
            print("Connection closed. Exiting.")


if __name__ == '__main__':
    client = Client(name='TableTennisAI')
    client.run()
