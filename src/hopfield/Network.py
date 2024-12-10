import numpy


class HopfieldNetwork:
	def __init__(self, input_size: int) -> None:
		self.__input_size: int = input_size
		self.__weight: numpy.ndarray = numpy.zeros((input_size, input_size))

	def memorize(self, patterns: numpy.ndarray) -> None:
		for pattern in patterns:
			self.__weight += numpy.outer(pattern, pattern)
		numpy.fill_diagonal(self.__weight, 0)

	def retrieve(self, stimulate: numpy.ndarray, max_iter: int = 10) -> numpy.ndarray:
		state: numpy.ndarray = stimulate.copy()
		for iteration in range(max_iter):
			previous_state: numpy.ndarray = state.copy()
			for i in range(self.__input_size):
				net_input = numpy.dot(self.__weight[i], state)
				state[i] = 1 if net_input > 0 else -1
			if numpy.array_equal(state, previous_state):
				break

		return state
