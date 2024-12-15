from math import ceil
from tkinter import Button, LabelFrame, Tk, messagebox
from tkinter.filedialog import askopenfilename

import numpy
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import figure

from hopfield.DataProcess import Dataset
from hopfield.Network import HopfieldNetwork

network = HopfieldNetwork(0)

window = Tk()

window.title("Hopfield")
window.resizable(False, False)
window.wm_title("Hopfield")
window.config(padx=10, pady=10)

visual_group = LabelFrame(padx=30, pady=30, border=0)
visual_group.pack(side="top", fill="both")

screen = figure(figsize=(4, 4))

canvas_data = FigureCanvasTkAgg(screen, master=visual_group)
canvas_data.get_tk_widget().pack(side="top", fill="both")

control_group = LabelFrame(padx=10, pady=10, border=0)
control_group.pack(side="bottom", fill="both")


def on_button_memorize_activate() -> None:
	pyplot.clf()

	file_path: str = askopenfilename()
	if file_path == "":
		messagebox.showwarning(title="Hopfield", message="You don't select any file.")
		return
	if file_path.find("txt") == -1:
		messagebox.showerror(title="Hopfield", message="File format should be .txt")

	dataset = Dataset(file_path)

	global network
	network = HopfieldNetwork(input_size=dataset[0].shape[0])
	for pattern in dataset:
		network.memorize(pattern)

	messagebox.showinfo(title="Hopfield", message="Memorized")


button_memorize = Button(control_group, text="Memorize", justify="center", command=on_button_memorize_activate)
button_memorize.pack(fill="x")


def on_button_recall_activate() -> None:
	file_path: str = askopenfilename()
	if file_path == "":
		messagebox.showwarning(title="Hopfield", message="You don't select any file.")
		return
	if file_path.find("txt") == -1:
		messagebox.showerror(title="Hopfield", message="File format should be .txt")

	dataset = Dataset(file_path)

	global network, screen
	num_column: int = ceil(len(dataset) / 3)
	for i in range(len(dataset)):
		memory: numpy.ndarray = network.recall(dataset[i]).reshape(dataset.pattern_shape)
		memory[memory <= 0] = 0
		ax = screen.add_subplot(3, num_column, i + 1)
		ax.imshow(memory, cmap="gray")
		ax.axis("off")

	pyplot.tight_layout()

	canvas_data.draw()


button_recall = Button(control_group, text="Recall", justify="center", command=on_button_recall_activate)
button_recall.pack(fill="x")

window.mainloop()
