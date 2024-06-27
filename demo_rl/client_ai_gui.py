import numpy as np
from client import Client, JOINTS, DEFAULT_PORT
import sys
import math
import threading
import tkinter as tk
from tkinter import ttk


# Create the main window
root = tk.Tk()
root.title("Joint Control")

# Function to update label when slider is moved


def update_label(slider, label):
    value = slider.get()
    label.config(text=f"{value:.2f}")


# Joint specifications
joints = [
    {"type": "Translation",
        "range": (-0.3, 0.3), "description": "Forward-Backward Slider. Positive Values are forward."},
    {"type": "Translation",
        "range": (-0.8, 0.8), "description": "Left-Right Slider. Positive Values are to the right."},
    {"type": "Rotation",
        "range": (-360, 360), "description": "Rotation around the vertical axis (Z)."},
    {"type": "Rotation", "range": (-math.pi/2, math.pi/2),
     "description": "Pitch of the first arm link."},
    {"type": "Rotation", "range": (-360, 360),
     "description": "Roll of the first arm link."},
    {"type": "Rotation", "range": (-3*math.pi/4, 3*math.pi/4),
     "description": "Pitch of the second arm link."},
    {"type": "Rotation", "range": (-360, 360),
     "description": "Roll of the second arm link."},
    {"type": "Rotation", "range": (-3*math.pi/4, 3*math.pi/4),
     "description": "Pitch of the third arm link."},
    {"type": "Rotation", "range": (-360, 360),
     "description": "Roll of the third arm link."},
    {"type": "Rotation", "range": (-3*math.pi/4, 3*math.pi/4),
     "description": "Pitch of the paddle."},
    {"type": "Rotation", "range": (-360, 360),
     "description": "Roll of the paddle."},
]

# Create and place sliders and labels
sliders = []
for i, joint in enumerate(joints):
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=i, column=0, sticky=(tk.W, tk.E))

    label = ttk.Label(frame, text=f"Joint {i} ({joint['type']}):")
    label.grid(row=0, column=0, sticky=tk.W)

    slider = tk.Scale(
        frame, from_=joint["range"][0], to=joint["range"][1], orient=tk.HORIZONTAL)
    slider.grid(row=0, column=1, sticky=(tk.W, tk.E))

    value_label = ttk.Label(frame, text="0.00")
    value_label.grid(row=0, column=2, sticky=tk.W)

    slider.config(command=lambda val, s=slider,
                  l=value_label: update_label(s, l))

    description = ttk.Label(frame, text=joint["description"], wraplength=300)
    description.grid(row=1, column=0, columnspan=3, sticky=tk.W)

    sliders.append(slider)


def run(cli):
    j = np.zeros((JOINTS,))
    j[2] = math.pi
    j[10] = math.pi / 2
    j[5] = math.pi / 2
    j[9] = math.pi / 4
    while True:
        state = cli.get_state()
        print(state)
        for i, slider in enumerate(sliders):
            j[i] = slider.get()

        # Agent
        bx = state[17]
        j[1] = bx

        # Send updated joint values to the server
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

    threading.Thread(target=run, args=(cli,)).start()
    # Start the GUI main loop in a separate thread
    root.mainloop()
    # Create the client and run the communication loop


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
