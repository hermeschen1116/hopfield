import numpy

from hopfield.DataProcess import Dataset


def signal_function(output: numpy.ndarray, state: numpy.ndarray) -> numpy.ndarray:
	if state.shape != output.shape:
		raise ValueError("signal_function: s and x should be in the same shape.")

	return numpy.ones_like(output) * (output > 0) + numpy.ones_like(output) * -1 * (output < 0) + state * (output == 0)


class HopfieldNetwork:
	def __init__(self, input_size: int) -> None:
		self.__input_size: int = input_size
		self.__weight: numpy.ndarray = numpy.zeros((input_size, input_size))
		self.__threshold: numpy.ndarray = numpy.zeros((input_size))

	def memorize(self, knowledges: Dataset) -> None:
		for knowledge in knowledges:
			self.__weight += numpy.outer(knowledge, knowledge)
		self.__weight /= self.__weight.shape[0]
		numpy.fill_diagonal(self.__weight, 0)

		self.__threshold = self.__weight.sum(0)

	def recall(self, stimulate: numpy.ndarray, max_iter: int = 10) -> numpy.ndarray:
		state: numpy.ndarray = stimulate.copy()
		for iteration in range(max_iter):
			previous_state: numpy.ndarray = state.copy()
			output: numpy.ndarray = numpy.sum(self.__weight * state, axis=1) - self.__threshold
			state = signal_function(output, state)
			if numpy.array_equal(state, previous_state):
				print(f"Recall Iteration: {iteration + 1}")
				break

		return state
