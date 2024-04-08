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

def search_lr():
    command_dict = {}
    for i, num_layers in enumerate([4]):
        name = f'num_layers_{num_layers}'
        command_dict[name] = f'python ../t2vec.py -num_layers {num_layers} -vocab_size 20000 -criterion_name "NLL" -checkpoint "/home/zlr/t2vec/data/num_layers_{num_layers}/checkpoint.pt" -knearestvocabs "/home/zlr/t2vec/data/porto-vocab-dist-cell100.h5"'
    grid_search(command_dict, 'search_num_layers')

search_lr()