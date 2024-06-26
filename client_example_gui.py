from client import Client, JOINTS, DEFAULT_PORT
import sys
import math
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Joint Control")

# Function to update label when slider is moved
def update_label(slider_var, label):
    value = slider_var.get()
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
slider_vars = []

for i, joint in enumerate(joints):
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=i, column=0, sticky=(tk.W, tk.E))

    label = ttk.Label(frame, text=f"Joint {i} ({joint['type']}):")
    label.grid(row=0, column=0, sticky=tk.W)

    slider_var = tk.DoubleVar()
    slider = tk.Scale(
        frame, from_=joint["range"][0], to=joint["range"][1], orient=tk.HORIZONTAL,
        resolution=0.01, variable=slider_var)
    slider.grid(row=0, column=1, sticky=(tk.W, tk.E))

    value_label = ttk.Label(frame, text="0.00")
    value_label.grid(row=0, column=2, sticky=tk.W)

    slider_var.trace_add('write', lambda name, index, mode, sv=slider_var, vl=value_label: update_label(sv, vl))

    description = ttk.Label(frame, text=joint["description"], wraplength=300)
    description.grid(row=1, column=0, columnspan=3, sticky=tk.W)

    sliders.append(slider)
    slider_vars.append(slider_var)

def run(cli):
    j = np.zeros((JOINTS,))
    j[2] = math.pi
    j[10] = math.pi
    j[5] = math.pi / 2
    j[9] = math.pi / 4
    while True:
        state = cli.get_state()
        #print(state)
        #diffferent prints:

        #pos of robot base:
        #print(f"Base (x,y):{state[0]},{state[1]}")

        print(f"Joint 10: {state[10]}")


        # Agent
        bx = state[17]

        #take slider info
        for i, slider_var in enumerate(slider_vars):
            j[i] = slider_var.get()
            if i==1 and j[i]==0:
                j[1] = bx
        #trying to get the paddle to be in the right position
        #if state[10]%(math.pi/2)==0: j[10]=0

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

if __name__ == '__main__':
    main()
