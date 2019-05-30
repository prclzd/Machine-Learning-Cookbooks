"""
This module provides some functions used in creating GUI for regression/model tree exploration.

Author:
    Hailiang Zhao
"""
import tkinter

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from src.statistical_learning.regression.tree import tree_reg


def re_draw(err_reduce_tol, min_instances):
    re_draw.fig.clf()
    re_draw.ax = re_draw.fig.add_subplot(111)
    if check_btn_var_model_tree.get():
        if min_instances < 2:
            min_instances = 2
        tree = tree_reg.create_tree(
            re_draw.raw_data,
            tree_reg.get_model_leaf,
            tree_reg.get_model_error,
            (err_reduce_tol, min_instances))
        y_hat = tree_reg.forecast(tree, re_draw.test_mat, tree_reg.evaluate_model_tree)
    else:
        tree = tree_reg.create_tree(
            re_draw.raw_data,
            ops=(err_reduce_tol, min_instances))
        y_hat = tree_reg.forecast(tree, re_draw.test_mat)
    re_draw.ax.scatter([re_draw.raw_data[:, 0]], [re_draw.raw_data[:, 1]], s=5)
    re_draw.ax.plot(re_draw.test_mat, y_hat, linewidth=2.0)
    re_draw.canvas.draw()


def get_inputs():
    # noinspection PyBroadException
    try:
        min_instances = int(entry_min_instances.get())
    except:
        min_instances = 10
        print('Enter integer for \'Min instances included\'')
        entry_min_instances.delete(0, tkinter.END)
        entry_min_instances.insert(0, '10')
    # noinspection PyBroadException
    try:
        err_reduce_tol = float(entry_err_reduce_tol.get())
    except:
        err_reduce_tol = 1.0
        print('Enter float for \'Error reduce tolerance\'')
        entry_err_reduce_tol.delete(0, tkinter.END)
        entry_err_reduce_tol.insert(0, '1.0')
    return min_instances, err_reduce_tol


def draw_new_tree():
    min_instances, err_reduce_tol = get_inputs()
    re_draw(err_reduce_tol, min_instances)


root = tkinter.Tk()
# tkinter.Label(root, text='Plot placeholder').grid(row=0, columnspan=3)
# create a matplotlib figure and assigns it to the global variable re_draw.fig
re_draw.fig = Figure(figsize=(5, 4), dpi=100)
# create a canvas widget
re_draw.canvas = FigureCanvasTkAgg(re_draw.fig, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# create the entry for the second parameter in ops: min_distances
tkinter.Label(root, text='Min instances included:').grid(row=1, column=0)
entry_min_instances = tkinter.Entry(root)
entry_min_instances.grid(row=1, column=1)
entry_min_instances.insert(0, '10')

# create the entry for the first parameter in ops: error_reduce_tol
tkinter.Label(root, text='Error reduce tolerance:').grid(row=2, column=0)
entry_err_reduce_tol = tkinter.Entry(root)
entry_err_reduce_tol.grid(row=2, column=1)
entry_err_reduce_tol.insert(0, '1.0')

# create redraw button
tkinter.Button(root, text='Redraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

# create check button for model tree
check_btn_var_model_tree = tkinter.IntVar()
check_btn = tkinter.Checkbutton(root, text='Model tree', variable=check_btn_var_model_tree)
check_btn.grid(row=3, column=0, columnspan=2)

# import data and draw on the GUI
re_draw.raw_data = np.mat(tree_reg.load_dataset('../../../../dataset/regression-examples/trees/sine.txt'))
re_draw.test_mat = np.arange(np.min(re_draw.raw_data[:, 0]), np.max(re_draw.raw_data[:, 0]), 0.01)
re_draw(1.0, 10)

root.mainloop()
