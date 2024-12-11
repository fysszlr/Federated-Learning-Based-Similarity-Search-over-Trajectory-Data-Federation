import libtmux
def grid_search(command_dict, tmux_name='search'):
    server = libtmux.Server()
    sess = server.new_session(tmux_name)

    for i, (name, command_) in enumerate(command_dict.items()):
        if i == 0:  
            window = sess.windows[-1]
        else:
            window = sess.new_window()
        window.rename_window(name)
        pane = window.panes[0]
        pane.send_keys(command_)

def search():
    command_dict = {}
    for i, weight in enumerate([0]):
        name = f'weight_{weight}'
        command_dict[name] = f'python /home/zlr/trajCl/train.py --dataset porto --moon_loss_weight {weight}'
    grid_search(command_dict, 'search_weight')

search()