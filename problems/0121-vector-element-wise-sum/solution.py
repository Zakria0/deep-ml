def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	if len(a) != len(b): 
		return -1

	v = []
	for i in range(len(a)):
		v.append(a[i] + b[i])
	
	return v