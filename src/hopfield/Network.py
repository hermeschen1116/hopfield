import numpy


def signal_function(s: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
	if x.shape != s.shape:
		raise ValueError("signal_function: s and x should be in the same shape.")

	return numpy.ones_like(s) * (s > 0) + numpy.ones(s) * -1 * (s < 0) + x * (s == 0)


class HopfieldNetwork:
	def __init__(self, input_size: int) -> None:
		self.__input_size: int = input_size
		self.__weight: numpy.ndarray = numpy.zeros((input_size, input_size))
		self.__threshold: numpy.ndarray = numpy.zeros((input_size))

	def memorize(self, patterns: numpy.ndarray) -> None:
		for pattern in patterns:
			self.__weight += numpy.outer(pattern, pattern)
		numpy.fill_diagonal(self.__weight, 0)
		self.__threshold = self.__weight.sum(0)

	def recall(self, stimulate: numpy.ndarray, max_iter: int = 10) -> numpy.ndarray:
		state: numpy.ndarray = stimulate.copy()
		for iteration in range(max_iter):
			previous_state: numpy.ndarray = state.copy()
			output: numpy.ndarray = state.dot(self.__weight) - self.__threshold
			state = signal_function(output, state)
			if numpy.array_equal(state, previous_state):
				break

		return state
