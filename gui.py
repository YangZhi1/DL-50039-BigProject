import PySimpleGUI as sg
import io
from PIL import Image
import deployment
import os

def get_nodes():
    filename = 'data/3980_edited.feat'
    all_nodes = []
    with open(filename, 'r') as f:
        cur_line = f.readline().split()
        while cur_line != []:
            all_nodes.append(cur_line[0])
            cur_line = f.readline().split()
    return all_nodes

all_nodes = get_nodes()

# Define the window's contents i.e. layout
layout = [
        [sg.Button('Show graph',enable_events=True, key='-GENERATE-', font='Helvetica 16'),
         sg.T('Enter the node indexes that are the scammers', key='lbl_a',font='consalo 14')
         , sg.Combo(all_nodes, key='node_index1', size=(10,1),pad=(10,10)), sg.Combo(all_nodes, key='node_index2', size=(10,1),pad=(10,10))],
        [sg.Image(key="-IMAGE-")], 
        [sg.Text('', size=(50, 10), key='-caption-', font='Helvetica 16')],
        [sg.Button('Exit')],
        ]

# Create the window
window = sg.Window('Scam prediction', layout, size=(800,700))

# Event loop
while True:
    event, values = window.read()
    node_index1 = values['node_index1']
    node_index2 = values['node_index2']
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == '-GENERATE-':
        deployment.get_graph(node_index1, node_index2)
        image = Image.open("graphimage.png")
        image.thumbnail((700, 600))
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        window["-IMAGE-"].update(data=bio.getvalue())
        

        window['-caption-'].update("Yellow: Not likely to be scam victim   \nGrey: Likely to be scam victim")

    
# Close the window i.e. release resource
window.close()
